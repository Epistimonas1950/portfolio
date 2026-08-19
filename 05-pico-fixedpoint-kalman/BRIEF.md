# 05 · Pico: a Fixed-Point Kalman Filter With a Stated Error Budget

> No FPU, 264 KB, hard real-time — every numerical choice has to be justified.

| | |
|---|---|
| **Effort** | 1–2 weeks |
| **Prerequisites** | None |
| **Hardware** | Raspberry Pi Pico (RP2040) or Pico 2 (RP2350) + MPU-6050 / LSM6DS3 |
| **Math** | Fixed-point error budgets, discretization, covariance positive-definiteness |
| **Status** | ☐ not started |

---

## Know the board before you plan the project

| | RP2040 (Pico) | RP2350 (Pico 2) |
|---|---|---|
| Core | 2× Cortex-M0+ @133 MHz | 2× Cortex-M33 @150 MHz |
| FPU | **none** | yes (single precision) |
| SRAM | 264 KB | 520 KB |
| LLM | never | never |

**On an RP2040 there is no floating-point unit.** Software float is available but slow, which is
exactly why this project is interesting: you write the filter in fixed point and you have to
*know* what that costs you. If you have a Pico 2, do it in fixed point anyway and use the FPU
build as your on-device reference.

Never propose training on this board. It runs inference and estimation, nothing else.

## The problem

An IMU gives you noisy accelerometer and gyroscope readings. The accelerometer is stable long-term
but noisy and corrupted by linear acceleration; the gyro is smooth short-term but drifts. A Kalman
filter fuses them into an orientation estimate. Then a small TensorFlow Lite Micro model
classifies gestures or keywords from the filtered stream.

That is a standard embedded project. **What makes it yours is the error budget** — stating, in
advance, how much precision each fixed-point operation costs and what the resulting drift will be,
then measuring it and showing you were right.

## The mathematics

**1. Q-format scaling.** Choose a fixed-point format `Qm.n` per quantity — angles, rates,
covariance entries, and gains all have very different dynamic ranges, and one global format is the
beginner's mistake. For each, bound the value range, pick `n` accordingly, and write down the
quantization step. The representation error per operation is `2⁻⁽ⁿ⁺¹⁾`; propagate it through the
filter recursion to get a predicted drift rate.

**2. Discretization.** The process model is continuous; the filter is discrete. State the
discretization (exact for a linear model, or first-order for small `dt`), and analyze how `dt`
interacts with numerical stability. Too large and the linearization breaks; too small and you
accumulate rounding error faster than you gain accuracy. **There is an optimal `dt` and you can
find it** — that is a genuinely nice result to have in a README.

**3. Joseph-form covariance update — the heart of the project.** The textbook update

```
P⁺ = (I − KH) P⁻
```

is algebraically correct and **numerically fragile**: in finite precision it does not preserve
symmetry or positive-definiteness, and the filter silently diverges. The Joseph form

```
P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ
```

is more expensive but preserves symmetry by construction and is far more robust to rounding.

**Demonstrate this.** Implement the naive form, show it losing positive-definiteness in fixed
point (log the eigenvalues or the diagonal going negative), then fix it with the Joseph form. A
short video or plot of a filter diverging and then not diverging is memorable, cheap to produce,
and shows you understand something most embedded developers hit by accident and never diagnose.

**4. Alternative worth mentioning:** square-root filtering (propagate a Cholesky factor of `P`
instead of `P` itself), which guarantees positive-definiteness and halves the dynamic range
requirement. Implement it if you have time; mention it either way.

## What to build

- [ ] I²C driver for the IMU, fixed sample rate with a hardware timer
- [ ] Double-precision reference implementation **on your laptop** (this is the ground truth)
- [ ] Fixed-point Kalman filter in C, Q-format documented per quantity
- [ ] Naive covariance update — captured diverging, on purpose
- [ ] Joseph-form update — the fix
- [ ] Error budget written **before** measurement, in the README
- [ ] Cycle counting and RAM high-water measurement
- [ ] TFLite Micro classifier (gesture or keyword) fed by the filtered stream
- [ ] Recorded dataset so laptop and board process identical input

## How it's measured

Record raw IMU data once, replay it through both implementations, and compare.

| Implementation | Drift over 60 s | Max abs error | Cycles/update | RAM peak |
|---|---|---|---|---|
| Float64 reference (laptop) | — | — | — | — |
| Fixed-point, naive update | | | | |
| Fixed-point, Joseph form | | | | |
| **Predicted from error budget** | | | — | — |

The last row is the point. Predicted vs. measured, agreeing within a stated factor.

## Interview claim

> I can put an estimator on a chip with no floating-point unit and tell you its numerical error
> budget in advance — not measure it afterwards and hope.

## Stack

C with the Pico SDK · CMSIS-DSP (fixed-point primitives and FFT) · TensorFlow Lite Micro ·
Python/NumPy for the reference implementation and analysis

## Suggested repo layout

```
pico-fixedpoint-kalman/
  README.md              <- error budget, then the divergence plot
  firmware/
    main.c
    kalman_fixed.c/h     Q-format documented at the top of the header
    kalman_joseph.c
    qformat.h            every format choice, with its range and step
    imu.c
    tflm/                model + wrapper
  reference/
    kalman_float.py      ground truth
    error_budget.py      predicted drift, executable
    compare.py
  data/
    imu_capture.csv
  results/
    divergence.png       naive vs Joseph
    predicted_vs_measured.png
```

## References

- [A big bang update for TensorFlow Lite for Microcontrollers (RP2040)](https://www.raspberrypi.com/news/a-big-bang-update-for-tensorflow-lite-for-microcontrollers/)
- [Pico4ML: RP2040-based platform for tiny machine learning](https://www.arducam.com/blog/pico4ml-an-rp2040-based-platform-for-tiny-machine-learning/)
- [TinyML on Raspberry Pi Pico with TFLite Micro and Arducam](https://arducam.medium.com/tinyml-machine-learning-on-raspberry-pi-pico-with-tensorflow-lite-micro-and-arducam-featuring-6c328fc0cd68)
- Grewal & Andrews, *Kalman Filtering: Theory and Practice* — the standard reference for Joseph
  form and square-root filtering.
- Bierman, G. J. *Factorization Methods for Discrete Sequential Estimation.*
