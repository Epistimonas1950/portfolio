# 05 · A Fixed-Point Kalman Filter With a Stated Error Budget

> A two-state attitude filter in integer arithmetic for a chip with no FPU, whose
> numerical failure points were written down as formulas **before** anything was run —
> and then measured, from the compiled C, to within a factor of 1.3 or a third of a bit.

**Status:** the filter, both covariance updates, the error budget and every sweep run
here today with `gcc` and `numpy` alone. There is no board attached, so the trace is
synthetic and there are no cycle or RAM numbers anywhere in this repo — see
[STATUS.md](STATUS.md).

---

## The headline result

Every prediction below comes from [`reference/error_budget.py`](reference/error_budget.py),
which computes them from the Q-format choices and the Riccati recursion **without ever
running the fixed-point filter**. Every measurement comes from
[`build/kfhost`](host/main.c), compiled with plain `gcc -O2 -Wall -Wextra -std=c11`.
Regenerate the whole table with `make host && make results`.

| Prediction, from the formats alone | Predicted | Measured | Agreement |
|---|---|---|---|
| Gain fractional bits below which the **naive** update's `P⁺₀₀` goes negative | **12.29 bits** | **12 bits** (13 survives) | **0.29 bits** |
| Naive `max‖P − Pᵀ‖`, swept over 14 gain precisions | 3.7e-8 … 7.8e-3 | 2.8e-9 … 1.3e-3 | **1.05–13×, median 1.7×** |
| Joseph `max‖P − Pᵀ‖`, at every precision from Q1.30 to Q1.10 | **exactly 0** | **exactly 0** | exact |
| Covariance fractional bits below which `Qd` rounds to zero | **23.4 bits** | survives 24, collapses 22 | brackets it |
| Sample-rate ceiling, Q1.24 covariance | **302 Hz** | 400 Hz | **1.32×** |
| Sample-rate ceiling, one global Q4.27 | **2416 Hz** | 3200 Hz | **1.32×** |
| Secular drift rate of the fixed-point error | **0 °/s** | 1.9e-6 °/s | 0.25× the floor over 60 s |
| Peak `\|θ_fixed − θ_float64\|` | ≤ 0.125° (bound) | 1.51e-3° | bound holds, 82× loose |

Raw numbers: [`results/error_budget.csv`](results/error_budget.csv),
[`results/comparison.csv`](results/comparison.csv),
[`results/gain_bits_sweep.csv`](results/gain_bits_sweep.csv),
[`results/cov_bits_sweep.csv`](results/cov_bits_sweep.csv),
[`results/dt_study.csv`](results/dt_study.csv).

**And the one the project exists for.** Same trace, same predict step, same gain, same
word lengths — only the covariance update differs
([`results/gain_bits_sweep.csv`](results/gain_bits_sweep.csv)):

| Kalman gain format | naive `min λ(P)` | naive `‖P−Pᵀ‖` | Joseph `min λ(P)` | Joseph `‖P−Pᵀ‖` |
|---|---|---|---|---|
| Q1.30 (nominal) | +2.035e-06 | 2.8e-09 | +2.034e-06 | **0** |
| Q1.24 | +2.034e-06 | 4.0e-07 | +2.034e-06 | **0** |
| Q1.18 | +2.035e-06 | 2.9e-05 | +2.035e-06 | **0** |
| Q1.13 | +2.031e-06 | 7.0e-04 | +2.037e-06 | **0** |
| **Q1.12** | **−1.562e-06** ← not a covariance | 1.3e-03 | **+2.033e-06** | **0** |
| Q1.10 | −1.562e-06 | 1.3e-03 | +2.020e-06 | **0** |

**What "goes indefinite" actually means here, precisely.** The naive covariance is
indefinite for **6 steps out of 12000** — the first 30 ms — and then recovers. That is
the honest version and it is the interesting one, because the damage does not recover
with it. The peak angle error against float64 is **0.272°** at t = 0.005 s, against the
Joseph form's 0.0064° (a factor of 42), and even restricting to **t > 1 s, long after
the covariance is positive definite again, the naive run's error is 0.0457° against
Joseph's 0.0064° — still 7× worse, for the remaining 59 seconds.** Six bad steps are
enough, because the state and bias estimates those steps produced are what the next
11994 gains are applied to. Joseph's symmetry residual, meanwhile, is not "small": it is
the integer zero, at every precision, for the reason given in
[the mathematics](#3-why-joseph-survives-and-the-textbook-form-does-not).

Filled in against the brief's own results table
([`results/comparison.csv`](results/comparison.csv); errors are against the float64
reference, because against truth they are dominated by sensor noise and say nothing
about arithmetic):

| Implementation | Drift over 60 s | Max abs error | Cycles/update | RAM peak |
|---|---|---|---|---|
| Float64 reference (laptop) | — | — | — | — |
| Fixed-point, naive, gain Q1.30 | 4.52e-4° | 1.51e-3° | *no board* | *no board* |
| Fixed-point, Joseph, gain Q1.30 | 4.52e-4° | 1.51e-3° | *no board* | *no board* |
| Fixed-point, naive, gain Q1.12 | 1.99e-3° | **0.272°** | *no board* | *no board* |
| Fixed-point, Joseph, gain Q1.12 | 2.42e-3° | 0.0064° | *no board* | *no board* |
| **Predicted, independent roundings** | 6.8e-5° | 6.8e-5° | — | — |
| **Predicted, ℓ1 upper bound** | 0.125° | 0.125° | — | — |

The cycles and RAM columns are empty and there is no estimate of them anywhere in this
repo. There is no board; a number I did not measure does not go in a table.

**Read the drift column with care — it makes the broken filter look better.** At Q1.12
the naive run's 60 s drift (1.99e-3°) is *lower* than Joseph's (2.42e-3°), and that is
not a typo. Once the covariance has been indefinite, the settled angle error stops being
a quality metric: the accelerometer drags θ back to the truth whatever P says, so both
filters end up within a few thousandths of a degree of each other and the ordering is
noise. The failure is in **P**, which is the object every subsequent gain is computed
from, and it shows up in the columns that look at P (`min_eigenvalue`,
`n_negative_steps`, `max_sym_resid`) and in the peak error, not in the settled one. A
results table that only carried drift would have missed this entirely, which is why the
covariance diagnostics are in it.

---

## The problem

An RP2040 has two Cortex-M0+ cores and **no floating-point unit**. A software `float`
multiply is 50–100 cycles where a `32×32→64` integer multiply is 1–4, so an estimator
that has to close a loop at 200 Hz is written in fixed point. Fixed point is not "float
with a shift": every quantity carries a statically chosen scale, and the quantities in
this filter span eleven orders of magnitude —

```
Kalman gain K1   up to  1.5e+1  1/s
gyro rate        up to  8.7e+0  rad/s
covariance Q11   down to 4.5e-8 rad^2/s^2   (the per-step process-noise increment)
```

— so one global format wide enough for the rates throws away, by construction, the bits
the covariance needs. That is the beginner's mistake the brief names, and this repo
prices it: **8.00× more predicted state error, and a sample-rate ceiling that falls from
19.3 kHz to 2.4 kHz**.

The second problem is the one most embedded developers hit by accident and never
diagnose. The textbook covariance update `P⁺ = (I − KH)P⁻` is algebraically correct and
numerically fragile: in finite precision it preserves neither symmetry nor positive
definiteness, and the filter degrades silently. The Joseph form does both by
construction. The point of this project is to say *at what word length* that matters,
in advance.

---

## The mathematics

### 1. The model, and why the discretisation is exact

State `x = [θ, b]ᵀ`: gravity-referenced angle (rad) and gyro bias (rad/s). The gyro
measures the true rate plus the bias:

```
d/dt θ = ω_meas − b + w_θ
d/dt b =           0 + w_b        w ~ N(0, diag(σ_g², σ_b²))
```

so `A = [[0, −1], [0, 0]]`, which is **nilpotent**: `A² = 0`, the matrix exponential
terminates after one term, and

```
F = exp(A dt) = I + A dt = [[1, −dt], [0, 1]]
```

is *exact for any dt*, not a first-order approximation. That matters for §5: any `dt`
dependence found there is a sampling or a rounding effect and cannot be a truncated
exponential. The discrete process noise is the Van Loan integral, also exact:

```
Qd = ∫₀^dt exp(As) Qc exp(As)ᵀ ds
   = [[σ_g² dt + σ_b² dt³/3,  −σ_b² dt²/2],
      [−σ_b² dt²/2,            σ_b² dt   ]]
```

At `dt = 5 ms` the off-diagonal is `−1.1e-10 rad²/s`, which is **0.12 of one Q1.30
ulp**: it rounds to zero, and `Qd` degenerates to `diag(Q00, Q11)`. That changes
`det(Qd)` by a relative 6e-6 and leaves it positive definite, so it is recorded as a
bounded approximation rather than pretended away.
→ [`firmware/kalman_fixed.h`](firmware/kalman_fixed.h), [`reference/kfparams.py`](reference/kfparams.py)

The measurement is the accelerometer's inclination estimate, `z = θ + v`, so `H = [1 0]`
and `R = σ_a²`. It is **scalar**, which is why the gain is one divide and not a 2×2
inverse — the single most consequential structural decision for a fixed-point port.

### 2. Q-format scaling, per quantity

Every format is derived from a hard bound, not from watching a trace. The table lives in
[`firmware/qformat.h`](firmware/qformat.h) with a justification per row; the shape of it:

| quantity | bound | format | step `2⁻ⁿ` | half-ulp `2⁻⁽ⁿ⁺¹⁾` |
|---|---|---|---|---|
| angle θ, innovation y | ≤ π rad | Q3.28 | 3.73e-09 | 1.86e-09 rad |
| gyro rate ω | ≤ 8.73 rad/s (±500 °/s FS) | Q4.27 | 7.45e-09 | 3.73e-09 rad/s |
| gyro bias b | ≤ 0.5 rad/s | Q1.30 | 9.31e-10 | 4.66e-10 rad/s |
| sample step dt | ≤ 0.5 s | Q0.31 | 4.66e-10 | 2.33e-10 s |
| covariance P, Q, R, S | `P00 ≤ 1 rad²` | Q1.30 | 9.31e-10 | 4.66e-10 rad² |
| Kalman gain K0 | `∈ [0,1)` identically | Q1.30 | 9.31e-10 | 4.66e-10 |
| Kalman gain **K1** | **≤ 25 1/s** | **Q5.26** | 1.49e-08 | 7.45e-09 1/s |

**`K1` is the one I got wrong first, and it is the sharpest argument in the repo for
per-quantity formats.** My first bound was Cauchy–Schwarz, `|K1| ≤ √(P11/P00)`, which I
read off as "about 0.5" — but that bound blows up as `P00 → 0` and says nothing. The
correct one maximises over the reachable set: with `|P10| ≤ √(P00 P11)` and
`S = P00 + R ≥ 2√(P00 R)`,

```
|K1| ≤ √(P00 P11)/(P00 + R) ≤ √(P11_max/R)/2 = √(0.25/1e-4)/2 = 25 1/s
```

attained near `P00 = R`, in the middle of the diffuse-prior transient. The float64
reference peaks at `|K1| = 15.4` at step 4, so the bound is tight to 1.6× and Q5.26 is
right. Built with Q1.30 instead, the gain saturated on **106 of 12000 steps** — and the
saturation counter is the only reason I found out. `K0` and `K1` are both "the Kalman
gain" and they need formats **four bits apart**.

**Two binding constraints, at opposite ends of the range.** The covariance must hold the
diffuse prior `P00(0) = 1 rad²` *and* resolve the per-step process noise
`Q11 = σ_b² dt = 4.5e-8`. That is `log₂(1/4.5e-8) = 24.4` bits of dynamic range before
any precision at all, out of the 31 magnitude bits an `int32` has. 6.6 bits of raw
margin — or **4.6**, the number [`firmware/qformat.h`](firmware/qformat.h) §2 states,
once you take out the integer bit that `P00 ≤ 1 rad²` needs and require `Qd` to be
resolved by at least 2 ulps rather than merely 1 —
which is exactly why the square-root form, which halves the dynamic-range requirement, is
the right answer if the specification tightens ([STATUS.md](STATUS.md)).

**Arithmetic.** All products go through `int64` and come back **saturated, never
wrapped**: an `int32` covariance entry that wraps from `+1.9` to `−1.9 rad²` is still a
perfectly ordinary integer, nothing downstream can tell, and the filter runs on with a
negative variance forever. Rounding is round-half-up rather than an arithmetic shift,
because a shift truncates toward `−∞` — a *biased* `−ulp/2` per operation, and a bias
inside a recursion is drift, which is the exact quantity being budgeted.
→ [`firmware/qformat.h`](firmware/qformat.h),
[`host/qformat_selftest.c`](host/qformat_selftest.c)

### 3. Why Joseph survives and the textbook form does not

With `H = [1 0]` the closed-loop matrix is lower triangular, `A = I − KH = [[a,0],[c,1]]`
with `a = 1 − K0`, `c = −K1`, and

```
P⁺₀₀ = a(a P00)                        + K0(K0 R)
P⁺₀₁ = c(a P00) + a P01                + K1(K0 R)
P⁺₁₀ = c(a P00) + a P10                + K1(K0 R)
P⁺₁₁ = c(c P00) + c P01 + c P10 + P11  + K1(K1 R)
```

Three properties, each load-bearing:

**Symmetry by construction.** `P⁺₀₁` and `P⁺₁₀` differ only in that one reads `P01`
where the other reads `P10`. Every other operand, shift and rounding is identical, so on
a symmetric input they are *the same int32* — not equal to within a rounding, identical.
`A P Aᵀ` is a congruence, and a congruence's `(i,j)` and `(j,i)` entries are the same
function of `(P, A)` with the indices exchanged. The naive form is not a congruence: its
`(0,1)` entry is `P01 − K0·P01` and its `(1,0)` entry is `P10 − K1·P00`, different
operands and different roundings. In exact arithmetic they agree, because
`K0 P01 = K1 P00 = P00 P01/S`; in fixed point the residual is **first order** in the
gain's representation error.

**A non-negative diagonal.** `a = 1 − K0 ∈ [0,1]`, so `a·(a·P00)` is a product of
non-negative numbers, `K0·(K0·R)` likewise, and round-to-nearest of a non-negative exact
value is non-negative. `P⁺₀₀` **cannot** go negative at any gain precision. The naive
`P⁺₀₀ = P00 − K0·P00` is a subtraction, and its true value is `P00 R/S` — during the
diffuse-prior transient (`P00 = 1 rad²`, `R = 1e-4 rad²`) four orders of magnitude below
either operand. An error `2⁻⁽ᵍ⁺¹⁾` in `K0` lands on it scaled by `P00`, so

```
P⁺₀₀ < 0   ⟺   2⁻⁽ᵍ⁺¹⁾ P00 > P00 R/S   ⟺   g < log₂(S_max/R) − 1 = 12.29 bits
```

**Measured: negative at Q1.12, positive at Q1.13.** The prediction is a closed form in
`S` and `R` and was written before the C compiled.

**The squares are chained, never formed.** `c·(c·P00)`, not `(c·c)·P00`;
`K1·(K1·R)`, not `(K1·K1)·R`. This is the trap I fell into first and it is worth the
warning: with `K1 ~ 1e-2`, `K1²` falls below one ulp of the gain format long before `K1`
does, so a coarse-gain build that expands the completed square silently **deletes the two
terms that make the result positive semi-definite** while keeping the indefinite cross
term `c(P01 + P10)`. The result is a Joseph filter that diverges *worse* than the naive
one — the right formula, the wrong arithmetic, and no symptom that points at the cause.

The cost of the guarantee is 9 multiplies against the naive form's 4.
→ [`firmware/kalman_joseph.c`](firmware/kalman_joseph.c),
[`firmware/kalman_fixed.c`](firmware/kalman_fixed.c)

### 4. The error budget: propagating `2⁻⁽ⁿ⁺¹⁾` through the recursion

Every routine rounds to nearest, so one operation into a `Qm.n` destination contributes
`|e| ≤ 2⁻⁽ⁿ⁺¹⁾`. Subtracting the fixed-point recursion from the float64 one cancels
everything except those roundings:

```
e_k = (I − K_k H) F e_{k−1} + w_k  =  A_k e_{k−1} + w_k
```

`A_k` is a contraction — spectral radius **0.9949** here — which is *why the budget
predicts a bounded floor rather than a ramp*. Two readings of `w_k` give two numbers:

- **variance propagation** `V_k = A_k V_{k−1} A_kᵀ + W_k`, each rounding independent and
  uniform, `ulp²/12`;
- **ℓ1 gain** `E = Σ_j |A_ss^j| |w|`, every rounding at its full half-ulp and perfectly
  aligned in time — a strict upper bound.

> The obvious worst case, `E_k = |A_k| E_{k−1} + |w_k|`, is **wrong** here and it looks
> right. Taking `|·|` entrywise destroys the sign structure that provides the damping:
> `|A|` has spectral radius above 1 even though `A`'s eigenvalues are 0.996 and 0.978, so
> that recursion diverges and predicts 10¹⁴ degrees. It is not a loose bound, it is not a
> bound. The ℓ1 gain — the summed absolute impulse response — is the correct worst case
> over all bounded rounding sequences.

**The channel I missed first, and it dominated.** The covariance is computed in fixed
point too, and its error feeds the gain:

```
dK0 = R·dP00/S²          dK1 = (dP10 − K1·dP00)/S
```

With `S ≈ 1e-4 rad²`, one ulp of error in `P10` becomes **10⁴ ulps of error in K1** — the
same divide that makes the scalar update cheap is a four-order-of-magnitude amplifier.
Leaving it out under-predicted the measurement by 70×. The covariance error needs its own
recursion, and it is the same matrix: at the optimal gain, the derivative of the Riccati
map with respect to `P` **is** the closed-loop congruence,

```
dP_k = A_k dP_{k−1} A_kᵀ + η_k        vec form: (A ⊗ A)
```

One matrix, two error channels. → [`reference/error_budget.py`](reference/error_budget.py)

**No secular drift, conditionally.** A bounded floor is only right while every correction
stays above its format's deadband. If `|K1·y|` ever fell below half an ulp of Q1.30, the
bias correction would round to zero, the bias state would freeze, and the angle would
then drift linearly. The budget prints the margin — **4.5e+5×** for the bias channel,
1.4e+5× for the angle — and predicts zero drift *because of it*. Measured slope over the
final 30 s: `1.9e-6 °/s`, i.e. `1.1e-4°` accumulated over 60 s against a `4.5e-4°` error
floor. It is a floor.

### 5. `dt`: the optimum is a property of the word length

`F` is exact, so shrinking `dt` cannot break a linearisation. What it does:

- **`dt` too large** — the gyro is held constant across the interval, an error of
  `(dt²/2)|ω̇|` per step. This is a *discretisation* error, present identically in
  float64, and the study measures both curves so it cannot be misattributed.
- **`dt` too small** — more steps per second, each injecting its roundings; and `Qd ∝ dt`,
  so past a certain rate the process noise falls below half a covariance ulp, rounds to
  zero, the filter stops admitting it can be wrong, and `P` collapses. That is a hard
  edge with a closed form:

```
Qd = σ² dt rounds to 0  ⟺  σ² dt < 2⁻⁽ⁿ⁺¹⁾  ⟺  fs > 2 σ² 2ⁿ
```

**A number format setting a sample rate.** Predicted 19.3 kHz at Q1.30, 2416 Hz for the
one-global-format Q4.27 build, 302 Hz at Q1.24; measured first indefinite covariance at
*never in the sweep*, 3200 Hz and 400 Hz — **1.32× in both measurable cases, the same
factor both times.**

Q1.24 is not a design point and I am not pretending it is: it is in the sweep **because**
`2σ²2ⁿ` puts its cliff at 302 Hz, inside a physically sensible sample-rate range where
the prediction can actually be tested. At the nominal Q1.30 the cliff is at 19.3 kHz and
no sane IMU rate reaches it, so the nominal build alone would have demonstrated nothing —
which is exactly what the first version of this sweep did, running only Q1.30 and Q4.27
and finding a monotone curve with no turning point. Choosing a word length so that a
closed-form threshold lands where it can be measured is what an error budget is *for*;
the formula and the 302 Hz came first, the run came second, and the test in
[`tests/test_discretisation.py`](tests/test_discretisation.py) asserts the prediction in
both directions so it cannot pass by always saying "it will break". RMS angle error against truth, 20 s at each rate
([`results/dt_study.csv`](results/dt_study.csv)):

| fs (Hz) | float64 | fixed, per-quantity Q | fixed, one global Q4.27 | fixed, Q1.24 covariance |
|---|---|---|---|---|
| 25 | 0.840° | 0.840° | 0.840° | 0.839° |
| 200 | 0.130° | 0.130° | 0.131° | 0.128° |
| 400 | 0.0865° | 0.0865° | 0.0876° | 0.0868° ← `λ_min` first negative |
| 1600 | 0.0555° | 0.0557° | 0.0556° | 0.0600° |
| 6400 | 0.0357° | 0.0361° | 0.0375° | 0.0540° ← best for this format |
| 12800 | 0.0297° | 0.0305° | 0.0348° | **2.118°** |

In float64, more samples are monotonically better across the whole sweep. In fixed point
with a Q1.24 covariance the returns stop, and at 12.8 kHz the filter is 71× worse than
the reference it was tracking. The optimal sample rate is not a property of the physics;
it is a property of the word length, and the budget names it before the sweep runs.
→ [`analysis/dt_study.py`](analysis/dt_study.py)

---

## Running it

No installation, no downloads, no network, no board. `gcc`, `python3` and `numpy`.

```bash
make host      # gcc -O2 -Wall -Wextra -std=c11 -> build/kfhost, build/qtest
make test      # 29 tests, ~9 s. Shells out to the compiled binary on purpose.
make results   # regenerates every CSV in results/  (~35 s)
make plots     # figures; needs matplotlib in a venv, see requirements.txt
make pico      # cross-compile for RP2040 -- explains why it cannot run here
```

The firmware in [`firmware/`](firmware) is portable C11 that includes only `<stdint.h>`
and contains no floating point, so the host build runs the **identical arithmetic** an
`arm-none-eabi` build would put on an RP2040 — same `int32_t` storage, same `int64_t`
intermediates, same shifts, same rounding. That is not a convenience; it is the reason
the whole numerical story is reproducible on a laptop with no hardware.

`make results` runs the error budget *before* the comparison, and the Makefile says so.
The predictions cannot be adjusted to fit what was measured because they are on disk
first.

### The tests that matter

Marked in the source with `=== THE TEST THAT MATTERS ===`. Each one fails if the
mathematics is wrong, not merely if the code crashed:

- **The naive update loses positive definiteness where Joseph does not**, at the gain
  precision the budget predicted in advance, on the same trace through the same predict
  step — and the naive form *survives* one bit higher, so the threshold is a prediction
  and not a direction. Joseph's symmetry residual is asserted to be exactly `0`, at 14
  precisions. → [`tests/test_filter.py`](tests/test_filter.py)
- **The predicted sample-rate ceiling is right in both directions**: below `2σ²2ⁿ` the
  covariance must stay positive definite at that word length, and above it must not.
  → [`tests/test_discretisation.py`](tests/test_discretisation.py)
- **The multiply saturates rather than wrapping.** `1.9 × 1.9` in Q1.30 must clamp to
  `+2.0`, not wrap to `−0.39`. A wrapped covariance is the worst failure available here
  because nothing downstream can detect it. → [`tests/test_qformat.py`](tests/test_qformat.py)
- **The fixed-point Joseph filter tracks float64 inside the budget** — under the ℓ1
  bound and above the independent-rounding estimate. Falling outside *either* end means
  the budget is wrong, which is the only claim this repo makes.
- **The C and Python constants are identical**, checked by asking the binary
  (`kfhost --params`). They are written out twice because this repo builds with bare
  `gcc` and runs with bare `python3`; a divergence would silently turn a numerical
  comparison into a modelling comparison. → [`tests/test_reference.py`](tests/test_reference.py)

---

## Results

Six CSVs, all committed, all regenerated by `make results`:

| File | What it holds |
|---|---|
| [`results/error_budget.csv`](results/error_budget.csv) | 86 predictions, written before any fixed-point run |
| [`results/comparison.csv`](results/comparison.csv) | predicted vs measured, the brief's table |
| [`results/gain_bits_sweep.csv`](results/gain_bits_sweep.csv) | 14 gain precisions × 2 update forms |
| [`results/cov_bits_sweep.csv`](results/cov_bits_sweep.csv) | 10 covariance precisions + the one-global-format build |
| [`results/dt_study.csv`](results/dt_study.csv) | 10 sample rates × 3 format sets + float64 |
| [`results/divergence_trace.csv`](results/divergence_trace.csv) | per-step `λ_min`, `‖P−Pᵀ‖`, error, both forms at Q1.12 |

`results/divergence.png` and `results/predicted_vs_measured.png` are **absent, not
empty**: matplotlib is not installed here and
[`analysis/plot_results.py`](analysis/plot_results.py) is the only file that imports it.
Every number the figures would show is in the CSVs above.

**Where the budget is loose, and why that is the finding.** The state-error *magnitude*
is the one prediction that is not tight. The independent-rounding estimate is 6.8e-5°;
the measurement is 4.5e-4°, **6.7× higher**. The diagnosis is in the data: the covariance
error was predicted at 2.4 ulps under a white-noise model and measured at 11.7 ulps,
while the strict ℓ1 bound puts it at 118 ulps. Rounding errors in a *deterministic*
recursion are neither white nor constant — a quantity sitting just above a grid point
rounds the same way every step, so the error behaves partly like a DC offset and picks
up the covariance loop's DC gain rather than its noise gain. Those two gains are
computable and they are **24.4** and **3.2** — a ratio of 7.6, which brackets the 4.8×
by which the white-noise model under-predicted the covariance error. The measurement
lands between the two models, close to their geometric mean. This is the same phenomenon as a
limit cycle in a fixed-point IIR filter, and I would rather report a bracket I can
explain than a point estimate I tuned. Note which predictions it does *not* affect: every
**threshold** in the headline table is a sign change or a first-order sensitivity, and
those are tight to a third of a bit or a factor of 1.3.

**The 1.05–13× range in the symmetry row is not scatter, and the 13× is at the nominal
format.** The predicted per-step injection has two terms,
`η = e_gain0·|P01| + e_gain1·P00 + 2·e_cov`, and which one dominates flips across the
sweep. At coarse gains the gain terms dominate, they are genuinely first-order, and the
agreement is **1.05–2.1×** at every precision from Q1.26 to Q1.13. At Q1.30 and Q1.28 the
gain terms have shrunk below the covariance-rounding floor `2·e_cov/K0 = 3.7e-8`, which
the model treats as always maximal and always aligned — and the measured residual there
is 3 covariance ulps, because it *is* an integer number of ulps and cannot be a fraction
of one. So the model is tight exactly where its dominant term is the one it models
correctly, and 13× loose where it has fallen back to a floor term that a
correlated-rounding argument (§4) already told us would be pessimistic. Same diagnosis,
different quantity.

---

## The limitation I volunteer first

**There is no board, and the trace is synthetic.** I did not record an IMU, so
[`data/imu_capture.csv`](data/imu_capture.csv) is four incommensurate sinusoids with a
seeded bias walk and Gaussian sensor noise, generated at exactly the noise densities the
filter's `Q` and `R` assume. That last part is deliberate and it cuts both ways: it makes
the filter correctly tuned by construction, so every discrepancy I measure is arithmetic
rather than a modelling mismatch — which is precisely what I wanted for *this*
measurement — but it also means the filter has never met vibration, linear acceleration
corrupting the accelerometer's inclination estimate, or temperature-dependent bias. A
real capture would not change a single line of the error budget, because the budget is a
property of the recursion and the formats and not of the data; it would change my right
to claim the filter is *tuned*. That is the first thing I would close.

**Second: no cycles, no RAM, and no estimate of either.** The brief asks for
cycles/update and RAM high-water. I cannot measure them without hardware, so those cells
are empty in the results table and there is no guess anywhere in this repo. What I will
say is a count and not a timing: Joseph is 9 multiplies against the naive form's 4. What
I will not say is what that costs on an M0+.

**Third: "identical arithmetic on device" is an argument, not a measurement.** The
firmware is `int32_t`/`int64_t` only, includes only `<stdint.h>`, and has no floating
point, so I believe the host build and an `arm-none-eabi` build produce bit-identical
results. I have not run it on an RP2040 and I have not run the cross-compiler. The two
places I would look first are the M0+'s lack of a 64-bit divide (my `q_div_f` promotes to
`int64`, which becomes a library call there) and any difference in how the compiler
implements the arithmetic right shift of a negative `int64`.

**Fourth: the naive form's failure needed a diffuse prior to appear.** At the nominal
Q1.30 gain, over 60 s, the naive filter is fine — its covariance is asymmetric by 3 ulps
and nothing else happens. The divergence needs `P00 ≫ R`, which here is the first second
after power-on with an unknown attitude. That is a real and common situation, not a
contrivance, but it is a *specific* one, and I would rather say so than imply the naive
form is always broken. The honest statement is the threshold: **below 12.3 gain
fractional bits, at this filter's conditioning**, and the conditioning is `S_max/R`,
which anyone can compute for their own filter from the formula in §3.

**Fifth: the square-root filter is not built.** It is the right answer to the
dynamic-range problem §2 identifies and the brief marks it optional. It is absent rather
than half-built, because Potter's update needs a fixed-point square root and that square
root carries its own error budget, which is the sort of thing this project exists to take
seriously rather than bolt on.

---

## References

- Grewal, M. S. & Andrews, A. P. *Kalman Filtering: Theory and Practice.* The standard
  reference for the Joseph form and for square-root filtering. The statement that
  `A P Aᵀ + K R Kᵀ` is the exact covariance of the estimator using **any** gain `K` —
  which is why a mis-rounded gain makes it suboptimal but never indefinite — is theirs.
- Bierman, G. J. *Factorization Methods for Discrete Sequential Estimation.* The
  factored alternatives (U-D, square-root) that this repo mentions in §2 and
  [STATUS.md](STATUS.md) but does not implement.

The brief also lists three RP2040 / TensorFlow Lite Micro references
([Raspberry Pi's TFLM announcement](https://www.raspberrypi.com/news/a-big-bang-update-for-tensorflow-lite-for-microcontrollers/),
[Pico4ML](https://www.arducam.com/blog/pico4ml-an-rp2040-based-platform-for-tiny-machine-learning/),
[TinyML on the Pico](https://arducam.medium.com/tinyml-machine-learning-on-raspberry-pi-pico-with-tensorflow-lite-micro-and-arducam-featuring-6c328fc0cd68)).
They belong to the classifier half of the brief, which is **not built** — see
[STATUS.md](STATUS.md).

Full brief: [BRIEF.md](BRIEF.md).
