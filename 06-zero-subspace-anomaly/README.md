# 06 · Streaming Subspace Tracking for Anomaly Detection

> An SVD that never stops, in 3.8 kB of state, written in C11 with no BLAS — because the
> board this is aimed at is armv6 and the framework approach does not run there at all.

**Status:** the C compiles with plain `gcc`, runs here, and produced every number below;
the numpy oracle it is checked against is exact. No Raspberry Pi Zero and no sensor were
attached to the machine this was built on, so the acquisition path and every on-board
figure are absent rather than estimated — see [STATUS.md](STATUS.md).

---

## The headline result

Detection of out-of-subspace anomalies, AUC, on labelled synthetic streams of 24 channels
whose normal operation lives near a rank-4 subspace. Identical anomalies in all three
scenarios by construction — same indices, same directions, same amplitude — so the only
variable is the structure of the *normal* class. Regenerate with `make results`; raw
numbers in [`results/auc.csv`](results/auc.csv).

| Method | AUC, 1 mode | AUC, 2 modes | AUC, 4 modes | State | Latency/sample | Runs on armv6? |
|---|---|---|---|---|---|---|
| Batch full SVD, laptop oracle | 0.9848 | 0.6434 | 0.5375 | `O(mn)`, 288–384 kB | — | — |
| Batch full SVD, all-normal (uses labels) | 0.9849 | 0.6473 | 0.5339 | `O(mn)` | — | — |
| **Incremental C, `r = 4`, reorth** | **0.9847** | 0.6480 | 0.5795 | **3 776 B** | ≈5.6 µs † | yes |
| Incremental C, no reorth | 0.9847 | 0.6480 | 0.5795 | 3 776 B | ≈5.6 µs † | yes |
| Incremental C, float32, reorth | 0.9847 | 0.6480 | 0.5795 | **1 948 B** | ≈5.3 µs † | yes |
| Incremental, rank from the gap criterion | 0.9847 | **0.9860** | 0.4937 | 3 776 B | ≈5.6 µs † | yes |
| TFLite autoencoder, Zero 2 W | | | | | | no |

† x86-64 host, `gcc -O2`, `clock()` over the 2 000-sample stream, from the
`us_per_sample_x86_host` column of [`results/c_vs_python.csv`](results/c_vs_python.csv).
The figure is for `r = 4`; the same column shows ≈6.4 µs at `r = 5` and ≈10.5 µs at
`r = 8`, which is the `O(mr + r³)` per-sample cost becoming visible. These four are the
only numbers in this README that move between runs — everything else is seeded and
reproduces exactly — so they are given to two significant figures and the CSV carries
whatever the last `make results` measured. **These are not Pi Zero
figures** and must not be read as such; no board was attached. The TFLite row is empty
because there is no hardware here to run it on — and on an *original* Zero there is no
armv6 wheel to run at all, which is the entire argument for this project.

The state figures are printed by the program itself (`tracker_bytes` in the summary), not
worked out by hand. 3 776 bytes is the whole steady-state footprint: basis, spectrum and
every scratch buffer, with no `malloc` in the per-sample path.

**And the number the project actually exists for** — orthogonality drift over 22 200
rank-one updates, `λ = 1`, from [`results/orthogonality_drift_summary.csv`](results/orthogonality_drift_summary.csv):

| Build | Repair | `‖UᵀU − I‖_F` initial | final | max | repairs | growth |
|---|---|---|---|---|---|---|
| C, double | **on** | 4.65e-15 | 1.74e-14 | 2.86e-14 | 35 | 3.8× |
| C, double | off | 4.65e-15 | **1.24e-13** | 1.46e-13 | 0 | **26.7×** |
| C, float32 | **on** | 4.21e-07 | 7.81e-06 | 1.40e-05 | 84 | 18.5× |
| C, float32 | off | 4.21e-07 | **2.63e-04** | 2.64e-04 | 0 | **625×** |
| numpy, double | off | 1.56e-15 | 1.11e-12 | 1.11e-12 | 0 | 712× |

Left alone, the basis stops being a basis, monotonically and silently. With the monitor
it is pinned at 1.3× the threshold, for 35 repairs in 22 200 updates.

The last row is worth a second look: the numpy tracker drifts an order of magnitude more
than the C running the same algorithm (1.11e-12 against 1.24e-13). That survives across four streams — 11×, 23×, 8.7×
and 2.7× ([`results/drift_c_vs_numpy.csv`](results/drift_c_vs_numpy.csv)) — so it is a real
difference and not one stream's luck. **I have not isolated which step accounts for it.**
The candidates are the small SVD (hand-written one-sided Jacobi in the C against LAPACK's
bidiagonalization in numpy, and Jacobi's high relative accuracy on small singular values is
the obvious suspect), the summation order in the truncation product, and the handling of
the near-zero singular value when the `ρ` guard fires. An eightfold spread across
streams argues against a single clean cause, and I would rather say that than pick the
flattering explanation.

And the correctness anchor
([`results/batch_agreement.csv`](results/batch_agreement.csv)): streaming 1 500 stationary
samples through Brand's update at `λ = 1` reproduces a full SVD of all 1 500 to a largest
principal angle of **9.1e-05 degrees** with singular values matching to **1.0e-09
relative** (C double); the numpy oracle reaches 1.1e-04 degrees and 2.6e-09, and the
float32 build 1.4e-01 degrees and 5.1e-06 — the price of the halved state.

---

## The problem

Multi-channel sensor data — vibration, audio, current draw. Normal operation lives near a
low-dimensional subspace; an anomaly is energy *orthogonal* to it. The batch solution is a
full SVD of the whole data matrix: `O(mn)` memory and `O(mn²)` work. On a 512 MB board
with a stream that never ends, that is not a slower option, it is not an option.

So track the subspace incrementally, in constant memory, in one pass.

The board matters here. A Raspberry Pi Zero is **armv6** (ARM11, 1 GHz, 512 MB). Almost
no modern ML wheel is built for armv6 — that is normally a nuisance, and here it is the
justification: this is a real detection problem solved with numerical linear algebra
compiled from source, on hardware where the framework route does not exist.

## The mathematics

### 1. Randomized range finding, and what `p` and `q` buy

The initial subspace comes from a Gaussian sketch of the warm-up block: `Ω ∈ ℝ^{n×(k+p)}`,
`Y = AΩ`, `Q = orth(Y)`. For `p ≥ 2` (Halko, Martinsson & Tropp 2011),

```
E ‖(I − QQᵀ)A‖_F  ≤  (1 + k/(p−1))^{1/2} · (Σ_{j>k} σ_j²)^{1/2}
E ‖(I − QQᵀ)A‖₂   ≤  (1 + √(k/(p−1))) σ_{k+1} + (e√(k+p)/p)(Σ_{j>k} σ_j²)^{1/2}
```

**`p` buys the constant and nothing else.** At `k = 8, p = 6` the Frobenius factor is
`(1 + 8/5)^{1/2} = 1.61`: in expectation, within 61 % of the best rank-8 projection that
exists. At `p = 1` the factor is undefined; at `p = 0` there is no guarantee, because `Y`
has exactly `k` columns and one draw nearly orthogonal to a singular direction loses it.
The deviation bounds in the same paper fail with probability decaying like `p^{-p}`, so
the handful of extra columns also buys a failure probability not worth thinking about.

**`q` buys everything `p` cannot.** Oversampling does not help a slowly decaying spectrum,
because the tail term is then genuinely large. Power iteration replaces `A` by
`B = (AAᵀ)^q A`, whose singular values are `σ_j^{2q+1}`, so the ratio of tail to signal is
raised to the `2q+1` before the bound is applied; with
`‖(I−QQᵀ)A‖ ≤ ‖(I−QQᵀ)B‖^{1/(2q+1)}` the whole bracket tends to `σ_{k+1}`, the optimum.

Measured on a 64×800 matrix with `σ_j = j⁻¹`, mean over 20 seeds, error of the sketch
truncated back to rank 8 as a multiple of the Eckart–Young floor
([`results/rangefinder.csv`](results/rangefinder.csv)):

| | `p = 0` | `p = 2` | `p = 6` | `p = 16` |
|---|---|---|---|---|
| `q = 0` | 1.537 | 1.413 | 1.241 | 1.093 |
| `q = 1` | 1.066 | 1.022 | 1.004 | 1.0002 |
| `q = 2` | 1.027 | 1.005 | 1.0002 | **1.0000** |

A slowly decaying spectrum is not incidental to that table: on a sharply decaying one even
`p = 0, q = 0` is already near-optimal and the experiment measures nothing.

**One thing that looks like a detail and is not.** Forming `(AAᵀ)^q AΩ` by repeated
multiplication is numerically hopeless — the product's singular values span `κ^{2q+1}`, so
at `q = 2, κ = 10³` the weak directions are below double precision and are rounded away.
Re-orthonormalizing between every application costs `O(m(k+p)²)` and restores them. This is
the difference between the power-iteration and subspace-iteration forms, and the one place
where a faithful transcription of the formula silently produces garbage.
→ [`src/rangefinder.h`](src/rangefinder.h), [`src/rangefinder.c`](src/rangefinder.c)

### 2. Brand's rank-one update

Given `A ≈ UΣVᵀ` and a new sample `a`, split it into what the model explains and what it
does not:

```
m = Uᵀa ,   p = a − Um ,   ρ = ‖p‖ ,   q = p/ρ
```

Then, **exactly**,

```
[ A  a ]  =  [ U  q ] [ Σ   m ] [ V  0 ]ᵀ
                      [ 0   ρ ] [ 0  1 ]
```

The middle matrix is `(r+1)×(r+1)`. Its SVD costs `O(r³)`; the updated factors are
`U ← [U q]U'` and `Σ ← Σ'`, truncated back to `r`. Per sample: `O(mr + r³)` work,
`O(mr + r²)` memory, one pass, no history re-read.

**`V` is deliberately not tracked.** It has one row per sample seen, so carrying it would
make memory grow with the length of the stream — precisely what "constant memory" forbids,
and the reason the algorithm is shaped this way. Nothing the detector needs lives in `V`.

Two things that look like details and are not:

- **Truncation requires sorted singular values.** One-sided Jacobi emits them in arbitrary
  order. Slicing the first `r` columns of an unsorted `U'` discards the *dominant*
  direction roughly one time in `r`. Nothing crashes, no value goes out of range, the
  scores stay plausible, and the tracked subspace is wrong.
- **The `ρ` guard.** When `a` lies in the subspace to machine precision, `ρ` is rounding
  noise and `q = p/ρ` is a random unit vector. The guard is `ρ ≤ √ε‖a‖`, applied by setting
  `ρ = 0` and `q = 0` rather than by branching: with `ρ = 0` the last row of the small
  matrix vanishes, so its top-`r` left singular vectors have zero last entry and the update
  degenerates *exactly* into the rank-preserving form. One code path, no second version to
  keep in sync.

→ [`src/incsvd.h`](src/incsvd.h), [`src/incsvd.c`](src/incsvd.c)

### 3. Reorthogonalization, and why Gram–Schmidt on `U` is the wrong repair

Repeated rank-one updates lose orthogonality of `U` in floating point. The failure is
silent in the strict sense: no exception, no NaN, no value out of range — `UUᵀ` merely
stops being a projection, so `‖a − UUᵀa‖` stops being a residual. So measure it, every
`check_every` samples at `O(mr²)`, and repair above a threshold of `100ε` (2.2e-14 in
double, 1.2e-05 in float, which is why one constant serves both builds).

The tempting one-liner is to run Gram–Schmidt on `U`. That is wrong: `UΣVᵀ` is a
*factorization*, and replacing `U` by the `Q` of `U = QR` changes what it factors unless
`Σ` moves too. The step that leaves the product invariant is

```
UΣ = QRΣ = Q(RΣ) = Q(Ũ S̃ Ṽᵀ)      ⟹      U ← QŨ ,   Σ ← S̃
```

a thin QR and an `r×r` SVD, `O(mr² + r³)`, amortised over `check_every` samples. Because
`R` is within `drift` of the identity, `S̃` is within `drift` of `Σ` — the correction is
tiny, which is exactly why skipping it looks harmless right up until it is not.
→ [`src/reorth.h`](src/reorth.h), [`src/reorth.c`](src/reorth.c)

**What the drift table does and does not show.** It shows the drift growing 26.7× (double)
and 625× (float32) over 22 200 updates without repair. It does *not* show the AUC changing: the "no reorth"
row of the headline table is identical to the "reorth" row to four decimals. Over 2 000
samples at 1e-13 orthogonality error, the detector cannot tell. That is the honest reading
and it is the reason the monitor exists: the error is real, it accumulates monotonically,
and it is invisible in the output metric until long after it has stopped being small.
A detector that runs for a week at 1 kHz sees 6×10⁸ updates, not 2×10³.

### 4. Forgetting

`Σ` is scaled by `λ` before each update, so a sample `k` updates old enters the current
second moment `UΣ²Uᵀ` with weight `λ^{2k}`. The effective window is the total weight:

```
N_eff = Σ_{k≥0} λ^{2k} = 1/(1 − λ²)      ⟹      λ = √(1 − 1/N_eff)
```

The factor of two is where this is usually got wrong: `λ` applied to `Σ` gives
`1/(1−λ²)`; the same symbol applied to a covariance or to `Σ²` gives `1/(1−λ)`. Both
appear in the literature. `N_eff = 400` gives `λ = 0.99875` and a half-life of 277 samples.

On a subspace that turns through 90° over 2 000 samples
([`results/forgetting.csv`](results/forgetting.csv)), largest principal angle to the true
final subspace, and the detector's own quantity — mean residual over the last 200 samples:

| `N_eff` | `λ` | angle to truth | mean tail residual |
|---|---|---|---|
| ∞ | 1.0 | **36.00°** | 0.1581 |
| 1000 | 0.99950 | 25.35° | 0.0726 |
| 400 | 0.99875 | 13.58° | 0.0186 |
| 200 | 0.99750 | 6.73° | 0.0040 |
| **100** | 0.99499 | 3.30° | **0.00208** |
| 50 | 0.98995 | **1.63°** | 0.00266 |

The two columns disagree, and that is the finding. The angle improves monotonically as the
window shortens — of course it does, a shorter window tracks better. The residual does not:
it bottoms out at `N_eff = 100` and rises again at 50, because a subspace estimated from
fewer effective samples is noisier. "Choose `λ` from the desired window length" is a real
tradeoff with an interior optimum, not a free parameter to turn down.
→ [`src/forget.h`](src/forget.h)

### 5. Rank selection

`r` is not hardcoded. Both criteria are computed from the warm-up spectrum and both are
reported ([`results/spectrum.csv`](results/spectrum.csv)):

| Stream | `σ₁ … σ₆` | energy ≥ 0.95 | largest gap |
|---|---|---|---|
| single mode | 141, 110, 84, 60, 1.4, 1.4 | **4** | **4** |
| rotating | 148, 115, 87, 60, 13.5, 8.9 | **4** | **4** |
| 2 modes | 153, 107, 69, 56, 43, 35 | **5** | **6** |
| 4 modes | 153, 63, 62, 52, 47, 46 | **10** | **1** |

They agree on the streams with a genuine cliff and disagree on the ones without, which is
the most useful thing about computing both. The energy criterion is the right one when the
downstream quantity is a residual energy — keeping 95 % bounds a normal sample's score by
5 % almost by definition — but it does not care where the spectrum breaks. The gap
criterion finds the cliff under a signal-plus-noise model and returns noise when there is
no cliff: on the four-mode stream it returns **1**, and the AUC that follows is 0.494.
→ [`src/rank.h`](src/rank.h)

### 6. The detector

```
s(a) = ‖a − UUᵀa‖² / ‖a‖²  ∈ [0, 1]
```

The normalisation is what makes the score comparable between a quiet stretch and a loud
one; without it the detector fires on every change in overall amplitude, which on a motor
is a change in load, not a fault. The numerator is `ρ²`, which the update already computed,
so the detector is free.

The threshold is the 0.99 empirical quantile of the scores over a 300-sample known-normal
calibration window — not `mean + 3σ`, because the null distribution is a scaled `χ²_{m−r}`
with a 32 % relative standard deviation and nothing like Gaussian. The only number chosen
by hand is a false-positive rate, which is a quantity an operator can reason about. On the
clean stream the realised rate over the 900 held-out samples is 2.0 % against a design
value of 1 % — the honest gap, and about what 300 calibration samples can pin down when
what is being estimated is a 1 % tail.
→ [`src/detect.h`](src/detect.h)

## Running it

No installation, no downloads, no network. gcc and numpy; nothing else.

```bash
make host      # gcc -O2 -Wall -Wextra -std=c11 -lm  ->  build/tracker, build/tracker32
make demo      # run the tracker on the labelled stream and print what it found
make test      # 52 tests, ~7 s; the C tests shell out to the real binary
make results   # regenerates every CSV in results/
make asan      # rebuild under AddressSanitizer + UBSan and exercise all three modes
make plots     # figures; needs matplotlib in a venv, see requirements.txt
make cross-note   # the armv6 cross-compile recipe, which is NOT run here
```

```bash
./build/tracker track --input data/anomalous.csv --output /tmp/scores.csv --lambda 0.99875
./build/tracker rangefind --input data/normal.csv --rank 8 --oversampling 6 --power-iters 2
./build/tracker selftest         # QR, Jacobi SVD, rank, quantile identities; exits non-zero on failure
```

Hand-written linear algebra with hand-managed scratch buffers is exactly the code whose
numbers should not be believed until a sanitizer has seen it. `make asan` rebuilds at `-O1`
with AddressSanitizer and UBSan and runs all three modes; it is clean, including the leak
check at exit.

The suite contains the assertions the project rests on, marked in the source with
`=== THE TEST THAT MATTERS ===`:

- **orthogonality drift is bounded with repair and grows without it** — *both halves*,
  because either alone is trivially satisfiable (a monitor that never fires passes the
  first; reorthonormalising every sample, which destroys the `O(mr)` cost that is the whole
  point, passes the second) —
  [`tests/test_subspace_tracking.py`](tests/test_subspace_tracking.py)
- **the incremental subspace matches the batch full SVD**, compared by principal angles and
  by `‖U_incU_incᵀ − U_batchU_batchᵀ‖_F`, never elementwise. A companion test demonstrates
  why: `U` and `UR` for orthogonal `R` differ elementwise by 0.84 and have a subspace
  distance of 1e-16, so an elementwise comparison tests the arbitrary choice of basis.
- **the range finder approaches the Eckart–Young optimum** as `p` and `q` grow, beats a
  no-oversampling baseline, never beats the optimum, and satisfies the HMT bound averaged
  over 40 draws (an expectation bound asserted on one draw is a flaky test) —
  [`tests/test_rangefinder.py`](tests/test_rangefinder.py)
- **the detector degrades on multiple operating modes**, and *recovers at the union rank* —
  [`tests/test_detector.py`](tests/test_detector.py)
- **forgetting tracks a rotating subspace where `λ = 1` does not**, and the residual has an
  interior minimum — [`tests/test_forgetting.py`](tests/test_forgetting.py)
- **the C agrees with the numpy oracle** on every stream —
  [`tests/test_c_implementation.py`](tests/test_c_implementation.py)

## Results

**The C against the oracle** ([`results/c_vs_python.csv`](results/c_vs_python.csv)), double
build, per-sample scores and the final tracked subspace:

| Stream | max relative score difference | subspace distance | largest principal angle |
|---|---|---|---|
| `normal.csv` | 1.2e-05 | 5.3e-07 | 1.8e-05° |
| `anomalous.csv` | 3.4e-06 | 2.4e-07 | 5.7e-06° |
| `multimode.csv` | 1.9e-06 | 3.8e-07 | 1.3e-05° |
| `rotating.csv` | 1.1e-05 | 2.2e-07 | 8.3e-06° |
| `manymode.csv` | 9.9e-03 | 5.8e-04 | 2.0e-02° |

The agreement is ~1e-6, not ~1e-15, and that is expected rather than a bug: the two
implementations seed different generators (PCG32 in the C, PCG64 in numpy), so their
initial sketches are different draws of the same randomized algorithm. This establishes
that the C implements the same algorithm to well within the algorithm's own approximation
error; it is not a bit-reproducibility claim. `manymode.csv` is three orders worse for a
specific reason: `σ₂ = 62.8` and `σ₃ = 62.5` are nearly equal, so the split between those
two directions is ill-conditioned and the two draws land on different bases for very nearly
the same subspace.

Everything else: [`results/auc.csv`](results/auc.csv),
[`results/roc.csv`](results/roc.csv) (TPR on a fixed 101-point FPR grid),
[`results/spectrum.csv`](results/spectrum.csv),
[`results/orthogonality_drift.csv`](results/orthogonality_drift.csv),
[`results/rangefinder.csv`](results/rangefinder.csv),
[`results/forgetting.csv`](results/forgetting.csv),
[`results/multimode_sweep.csv`](results/multimode_sweep.csv),
[`results/scores_anomalous_c.csv`](results/scores_anomalous_c.csv) (the C's raw per-sample
output). `make plots` draws them; matplotlib is not installed here, so the figures are not
committed and every number lives in a CSV instead.

### Why synthetic data

Real capture is impossible on this machine — there is no board and no sensor — and
[`oracle/generate_data.py`](oracle/generate_data.py) says so in its first paragraph rather
than dressing the generator up as a recording. What it buys, beyond necessity, is that the
null distribution is known exactly: normal residual energy is `σ²χ²_{m−r}`, so a detector
failure can be told apart from a data surprise. On a real motor I would not know the true
rank, and I could not.

## The limitation I volunteer first

**I shipped the wrong version of this limitation first, and the measurement corrected me.**
BRIEF.md says a single subspace blurs several operating modes and the detector degrades. It
does — on the two-mode stream the AUC falls from 0.985 to 0.648. But changing one thing,
the rank criterion from energy(0.95) to the singular-value gap, takes it straight back to
**0.986**. Two overlapping rank-4 modes span a rank-6 union; at `r = 6` the subspace
contains both and the detector works. What failed there was rank selection wearing the
costume of a modelling failure, and shipping only that scenario would have been a
comfortable, wrong story.

So here is the four-mode stream, and the whole surface
([`results/multimode_sweep.csv`](results/multimode_sweep.csv)), AUC by tracked rank and
effective window:

| `r` | complement dim | `N_eff = 400` | `N_eff = 50` | batch oracle at same `r` |
|---|---|---|---|---|
| 4 | 20 | 0.486 | 0.493 | 0.505 |
| 8 | 16 | 0.580 | 0.683 | 0.537 |
| 10 | 14 | 0.659 | **0.850** | 0.662 |
| **13** | 11 | **0.978** | 0.965 | 0.979 |
| 16 | 8 | 0.976 | 0.968 | 0.979 |

Four rank-4 modes sharing one direction span a rank-13 union, and at `r = 13` both the
tracker and the batch oracle come back to 0.98. **The subspace model survives multiple
modes; what does not survive is automatic rank selection.** Neither criterion in this repo
finds 13 from that spectrum — energy(0.95) returns 10 and the gap criterion returns 1 —
because a union-of-modes spectrum has no cliff to find. That is the honest statement, and
it is narrower and more useful than "subspace methods fail on multimodal data".

Both escape routes cost something, and the table prices them:

- **Raise `r`.** The detector measures energy in the orthogonal complement, whose dimension
  is `m − r`. Going from `r = 4` to `r = 13` shrinks it from 20 to 11, and the unimodal AUC
  falls from 0.9847 to 0.9826 — small here, and at `r → m` every score is zero.
- **Shorten the window.** At `r = 10`, dropping `N_eff` from 400 to 50 lifts the AUC from
  0.659 to 0.850, because a window shorter than the 100-sample dwell lets the tracker lock
  onto the mode currently running. But it needs the dwell time in advance — which is the
  mode structure you were claiming not to know — and §4 above shows the residual rising
  again once the window gets too short.

The fix that does not cost either is a **mixture of subspaces**: one rank-4 model per mode,
recovering a 20-dimensional complement instead of an 11-dimensional one, at the price of
identifying modes online. It is not built. Also not built and worth saying: the threshold is
calibrated once and never revisited, so a drift in the healthy noise level moves the
false-positive rate and nothing notices; and rank is fixed after warm-up rather than
adapted.

**Second: no board, no sensor, no capture.** Every latency figure here is x86-64. The
memory figures are real but they are the program's own accounting of its heap, not an
`/proc/self/status` reading on a Zero. The TFLite comparison row is empty because there is
nothing here to run it on. All of this is itemised in [STATUS.md](STATUS.md).

**Third: I am implementing published algorithms.** Brand's update and the HMT range finder
are both in the literature and I am not claiming otherwise. The claim is that I implemented
them from the mathematics with no library underneath, can state the error bound and its
hypotheses, and went looking for where the numerics break — which is what the drift study,
the `ρ` guard and the rank-criterion disagreement are.

## References

- Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011). [*Finding Structure with Randomness:
  Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions*](https://arxiv.org/abs/0909.4061) (arXiv:0909.4061; SIAM Review
  53(2):217–288, [doi:10.1137/090771806](https://doi.org/10.1137/090771806)) — the
  range-finder bound quoted in §1, with the oversampling and power-iteration analysis.
- Brand, M. (2006). [*Fast low-rank modifications of the thin singular value
  decomposition*](https://doi.org/10.1016/j.laa.2005.07.021), Linear Algebra and its
  Applications 415(1):20–30 — the rank-one update of §2.
- Balzano, L., Chi, Y. & Lu, Y. M. *Streaming PCA and Subspace Tracking: The Missing Data
  Case* — survey of the field; several preprint copies circulate, search by title.

Full brief: [BRIEF.md](BRIEF.md).
