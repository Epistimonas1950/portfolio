# 01 · Activation-Aware Low-Rank Compression by Whitened SVD

> Truncated SVD is optimal in a norm nobody downstream cares about. One Cholesky
> factor moves the optimality into the norm that changes the layer's output — and
> this repo measures exactly how much that is worth, and exactly when it is worth
> nothing.

**Status:** the mathematics runs, is tested, and produces every number below on
synthetic layers with numpy alone. The real-model extension — forward hooks,
WikiText-2, perplexity, GGUF export — is **not built**, and is listed as not built.
See [STATUS.md](STATUS.md).

---

## The headline result

Relative activation-weighted error `‖(W − Ŵ)X‖_F / ‖WX‖_F` on **held-out**
activations, mean over 5 seeds, on 256×256 synthetic layers whose activation
covariance has condition number ≈ 2.8·10⁵. Both methods get the identical rank, so
identical parameter count, so identical compression. Regenerate with `make results`;
raw numbers in [`results/pareto.csv`](results/pareto.csv).

| Compression | rank | Plain truncated SVD | Whitened SVD | Margin |
|---|---|---|---|---|
| 16.00× | 8 | 0.9304 | 0.8058 | 1.15× |
| 8.00× | 16 | 0.8639 | 0.6585 | 1.31× |
| 4.00× | 32 | 0.7489 | 0.4418 | 1.70× |
| 2.67× | 48 | 0.6454 | 0.2991 | 2.16× |
| **2.00×** | 64 | **0.5540** | **0.2025** | **2.74×** |
| 1.33× | 96 | 0.4107 | 0.0919 | 4.47× |
| 1.14× | 112 | 0.3515 | 0.0618 | 5.68× |

**At 2× compression the whitened factorization has 2.74× less activation-weighted
error than truncated SVD at the same parameter count.** The margin grows as the rank
grows, because plain SVD's error is dominated by directions it discards *regardless*
of rank, while the whitened error tracks the activation spectrum and falls away with
it.

And the control, which is the reason to believe the table above. Same code, same
ranks, same seeds — but on **isotropic** activations, where `M = XXᵀ` is a multiple of
the identity, `S ∝ I`, and the two objectives are the same objective:

| Compression | Plain SVD | Whitened SVD | Margin (held-out) | Margin (in-sample) |
|---|---|---|---|---|
| 8.00× | 0.8630 | 0.8837 | 0.98× | 1.04× |
| 4.00× | 0.7462 | 0.7792 | 0.96× | 1.06× |
| 2.00× | 0.5547 | 0.6027 | **0.92×** | 1.12× |
| 1.14× | 0.3512 | 0.4058 | **0.87×** | 1.20× |

The advantage does not merely shrink — **it inverts.** What is left of it in-sample
(1.04×–1.20×) is whitening fitting the Wishart sampling noise in `M`, and on held-out
activations that costs 8–13%. This is the mirror test doing its job: had the
anisotropic margins survived here, the 2.74× above would have been an artefact of
something other than the advertised mathematics.

---

## The problem

Factor every weight matrix `W ≈ AB` with `A: m×r`, `B: r×n`, and `mn` stored
parameters become `r(m+n)` — a saving whenever `r < mn/(m+n)`, i.e. `r < 128` for a
square 256×256 layer. The obvious method is truncated SVD, justified by Eckart–Young:
the rank-`r` truncation minimizes `‖W − Ŵ‖_F` over all rank-`r` matrices.

That justification is the trap, and it is the whole project. Eckart–Young optimality
is in the **unweighted** Frobenius norm. A layer's job is to produce `WX`, so the
error that changes the model's behaviour is

```
‖(W − Ŵ) X‖_F
```

where `X` holds the activations the layer actually sees. Truncating the smallest
singular values of `W` discards directions that may be heavily excited by real input
while keeping directions the model never visits. Plain SVD compression of LLMs is
reported to collapse for exactly this reason, and the isotropic control above is what
that failure looks like when you remove the cause.

## The mathematics

### 1. Whitening puts Eckart–Young back in charge

Collect activations `X` (n × n_samples) and form the second moment `M = XXᵀ`. Write
`E = W − Ŵ` and expand the trace:

```
‖(W − Ŵ)X‖²_F  =  tr(E X Xᵀ Eᵀ)  =  tr(E M Eᵀ)
```

`M` is symmetric positive semidefinite, so it has a Cholesky factor `M = S Sᵀ`, and

```
tr(E S Sᵀ Eᵀ)  =  ‖E S‖²_F        ⟹        ‖(W − Ŵ)X‖_F  =  ‖(W − Ŵ)S‖_F     (1)
```

The right-hand side is an *unweighted* Frobenius norm. Since `S` is invertible,
`Ŵ ↦ ŴS` maps the rank-`r` matrices bijectively onto themselves, so minimizing (1)
over rank-`r` `Ŵ` is exactly the Eckart–Young problem for `WS`:

```
WS = UΣVᵀ ,   truncate to rank r ,   Ŵ = U_r Σ_r V_rᵀ S⁻¹                        (2)
```

and that `Ŵ` is the **global** minimizer of the activation-weighted error over all
rank-`r` matrices — not a heuristic. (Exactly so at `λ = 0`; §2 below says precisely
which objective the ridge substitutes, and every number in this repo uses `λ > 0`.) Note what this says about plain truncated SVD: it
is *also* a global minimizer, of a different problem. Both methods are exactly optimal;
only one of them is optimal for the question. → [`src/whiten.py`](src/whiten.py),
[`src/factorize.py`](src/factorize.py)

`S⁻¹` is never formed. `B = V_rᵀS⁻¹` is the solution of `SᵀBᵀ = V_r` with `Sᵀ` upper
triangular, so one back substitution does it in `O(n²r)`.

### 2. The ridge is an interpolation, not a fudge

`M` is routinely near-singular — directions the calibration set barely excites give
near-zero eigenvalues and `S⁻¹` amplifies them by the inverse of their square roots.
With fewer calibration samples than input channels `M` is singular *by construction*
and the Cholesky must fail rather than return a silently amplified factor (it does;
[`tests/test_whiten.py`](tests/test_whiten.py) asserts it).

The ridge `M + λI = SSᵀ` has an exact reading. Substituting into (1):

```
‖E S‖²_F  =  tr(E (M + λI) Eᵀ)  =  ‖E X‖²_F  +  λ ‖E‖²_F                        (3)
```

**The ridge is a convex interpolation between the activation-weighted objective and
the plain one**, and `λ → ∞` recovers plain truncated SVD exactly, because
`S → √λ · I`. So `λ` is a dial between the two methods in the table above, which is
also why the isotropic case shows no advantage — there `M` is already a multiple of
the identity, so every `λ` gives the same answer.

`λ = ratio · mean(diag M)` is dimensionless, so the same value transfers across layers
whose activation energies differ by orders of magnitude. The measured behaviour at 2×
compression ([`results/pareto.csv`](results/pareto.csv)):

| ridge ratio | in-sample | held-out | half-shifted `X` | fully shifted `X` |
|---|---|---|---|---|
| 10⁻⁶ | 0.1689 | 0.2025 | 0.4854 | 0.7046 |
| 10⁻⁴ | 0.1689 | 0.2025 | 0.4854 | 0.7046 |
| 10⁻² | 0.1689 | 0.2024 | 0.4838 | 0.7025 |
| 1 | 0.2644 | 0.2913 | **0.4541** | **0.6044** |
| plain SVD (λ = ∞) | 0.5534 | 0.5540 | 0.5525 | 0.5553 |

Two things worth saying out loud. First, over six decades the ridge is free — it buys
a factorization and costs nothing — and then it degrades continuously into the
baseline, which is equation (3) being true. Second, **the best `λ` depends on how far
the test distribution has moved**: at `λ = 1` the method is worse in-domain and better
under shift, because (3) is literally mixing in the `X`-agnostic objective. That is a
bias–variance dial, and reading it off the identity rather than off a grid search is
the point.

### 3. Spectra: why whitening makes a layer compressible at all

`W` and `WS` are the same matrix in different coordinates, and their spectra are not
remotely the same shape. The scalar that captures it is the stable rank
`‖A‖²_F / ‖A‖²₂`, the effective number of energy-carrying directions
([`results/spectra.csv`](results/spectra.csv)):

| Layer | shape | cond target | cond(M) | cond(M + λI) | stable rank `W` | stable rank `WS` |
|---|---|---|---|---|---|---|
| L0 | 128×256 | 10⁶ | 1.05e+08 | 5.64e+07 | 27.8 | **2.09** |
| L1 | 192×192 | 10⁵ | 7.78e+06 | 7.21e+06 | 41.6 | **2.02** |
| L2 | 256×128 | 10⁴ | 1.13e+06 | 1.11e+06 | 27.8 | **1.83** |
| L3 | 128×128 | 10³ | 8.41e+04 | 8.40e+04 | 27.8 | **2.04** |
| L4 | 256×256 | 10⁵ | 7.30e+06 | 6.80e+06 | 55.3 | **2.73** |
| L5 | 64×320 | 10² | 5.77e+04 | 5.77e+04 | 14.1 | **1.82** |

For L4: `σ_64/σ_1` is 0.566 for `W` and 0.0359 for `WS`. Plain truncated SVD at rank
64 is throwing away directions still carrying 57% of the top singular value.
Whitened SVD is throwing away directions carrying 3.6%. That is the whole method in
one line. → [`analysis/spectra.py`](analysis/spectra.py)

### 4. Rank allocation under a global budget

Uniform compression across layers is leaving accuracy on the table — layers differ in
shape, so a unit of rank costs a different number of parameters, and in spectral
decay, so it buys a different amount of error reduction. Given a budget `B`:

```
minimize   Σ_ℓ L_ℓ(r_ℓ)     subject to   Σ_ℓ r_ℓ (m_ℓ + n_ℓ)  ≤  B ,
L_ℓ(r) = Σ_{i>r} σ²_{ℓ,i}   from the whitened spectrum
```

`L_ℓ` is not a proxy. By identity (1), at zero ridge it **is**
`‖(W_ℓ − Ŵ_ℓ)X_ℓ‖²_F` for the rank-`r` whitened optimum. Measured across all 24
allocations, `proxy / measured` lands in [1.000008, 1.000057] — the residual is the
`λ‖E‖²` term of (3) at `ridge = 10⁻⁶`, and it has the sign the identity predicts.
See [`results/allocation.csv`](results/allocation.csv), and the assertion in
[`tests/test_allocate.py`](tests/test_allocate.py). It is a proxy only for what a
*stack* does end to end, where each layer's error becomes the next layer's input
perturbation.

Four solvers, on a six-layer stack with deliberately heterogeneous shapes and
conditioning. Excess loss over the exact optimum, at each budget:

| Budget (of dense) | uniform | greedy | Lagrangian | knapsack DP |
|---|---|---|---|---|
| 10% | +23.0% | +0.00% | +0.00% | 0 |
| 15% | +28.6% | +0.00% | +0.00% | 0 |
| 20% | +30.6% | +0.36% | +0.36% | 0 |
| 30% | +42.2% | +0.00% | +0.78% | 0 |
| 40% | +53.0% | +0.00% | +0.00% | 0 |
| 50% | +63.4% | +0.39% | +0.39% | 0 |

In stack relative error at a 20% budget (≈5.0× compression) that is 0.2355 for
uniform against 0.2061 for the optimum.

**The interesting result is that greedy and Lagrangian essentially tie with the exact
optimum, and that is a prediction rather than a coincidence.** Because `σ` is sorted
descending, the marginal gain of layer `ℓ`'s `(r+1)`-th component, `σ²_{ℓ,r+1}`, is
non-increasing in `r`. So this is a separable allocation problem with convex per-layer
losses, and for that class incremental greedy is optimal up to the last partial item.
The residual gaps are pure integrality:

* **Lagrangian** bisects a multiplier `μ` until the budget binds, and at the solution
  every layer sits where its marginal loss per parameter equals `μ` — the rates are
  equalized, which is what optimality of a separable allocation looks like. But the
  achievable costs form a discrete lattice, so the smallest feasible `μ` usually
  leaves budget unspent: 256 of 61 440 parameters at the 30% budget, which is the
  entire +0.78%.
* **Greedy** spends leftover budget on whatever still fits, so it ties the DP more
  often; where it does not (20%, 50%), it has stranded 128 parameters that the DP
  spends by shifting one rank between two layers of different width.
* **The DP is exact on its grid** — every integer rank, with the budget measured in
  units of the gcd of the per-rank costs so nothing is rounded — and not exact outside
  it.

A candidate who reports a manufactured disagreement between these three has not
looked at the convexity. → [`src/allocate.py`](src/allocate.py),
[`analysis/allocation_compare.py`](analysis/allocation_compare.py)

### 5. Composing with quantization: where the crossover actually sits

Low-rank and quantization are alternative ways to spend the same budget — rank against
bits — so the honest question is not "does whitening help" but "given a compression
target, what should I buy". Storing a factored, quantized layer costs

```
r (m + n) b  +  (m + r) · 16   bits          [payload + per-row fp16 scales]
```

against `m n · 16` dense, so a target compression picks a *curve* in the `(r, b)` plane
rather than a configuration. → [`src/compose.py`](src/compose.py)

**The measurement has to be set up carefully, and the obvious setup is rigged.**
Quantization offers only a discrete ladder of compressions — one rung per integer
bit-width, here 1.98×, 2.64×, 3.94×, 5.22×, 7.76× and nothing in between — while rank is
nearly continuous. Choosing round targets like "6×" forces quantization to the next rung
*up* (2 bits, 7.76×) and then compares a 7.76×-compressed quantized layer against a
6×-compressed factored one. My first version did exactly that and produced a crossover
one rung too early. The sweep is now anchored on the quantizer's own achievable rungs:
at each bit-width the low-rank family is handed exactly the compression the quantizer
achieved, and both arms return their best configuration at that budget.

Relative activation-weighted error, mean over 3 seeds, 256×256 layers
([`results/composition.csv`](results/composition.csv)):

| Achieved | quant. | **isotropic** | | **anisotropic** | | **strongly spiked** | |
|---|---|---|---|---|---|---|---|
| | bits | quant | low-rank | quant | low-rank | quant | low-rank |
| 1.98× | 8 | **0.0060** | 0.1515 | **0.0026** | 0.0325 | **0.0011** | 0.0161 |
| 2.64× | 6 | **0.0246** | 0.2127 | **0.0105** | 0.0546 | **0.0044** | 0.0171 |
| 3.94× | 4 | **0.1087** | 0.2925 | **0.0463** | 0.1052 | **0.0195** | 0.0304 |
| 5.22× | 3 | **0.2548** | 0.3814 | **0.1086** | 0.1762 | **0.0460** | 0.0512 |
| 7.76× | 2 | 0.7231 | **0.5076** | 0.3293 | **0.2239** | 0.1393 | **0.0725** |

**The crossover sits between 5.22× and 7.76× — that is, between 3-bit and 2-bit
quantization — and it lands in the same bracket in all four activation spectra**, from
the isotropic control through `cond(XXᵀ) = 10⁷` with eight spiked channels. The *margin*
moves a great deal with the spectrum (at 5.22× quantization wins by 1.50× on isotropic
activations and by only 1.11× on strongly spiked ones, because anisotropy helps low-rank
more than it helps quantization), but the crossing point does not.

Two things fall out that I did not expect:

- **Neither method wins alone.** Every winning low-rank-family configuration in that
  table is a *composed* one — factored *and* quantized, e.g. `r = 63, b = 4`. Factors
  left in fp16 never win at any compression above 2×. The two methods are not rivals
  with a crossover so much as two knobs on one control surface.
- **Quantization runs out of road before low-rank does.** Two bits is the practical
  floor for a uniform symmetric grid, which caps quantization-only at 7.76× on this
  layer. Past that the comparison is not close — it is unavailable, and rank is the only
  remaining budget to spend.

**Order.** Low-rank first, then quantize, is the composition that composes. The two
factors do not see the same input: `B` multiplies the layer activations `X`, while `A`
multiplies `B̂X`, the output of the factor already decided — so `A`'s Hessian is over the
`r` propagated directions, not the `n` input ones. Quantizing `A` against an isotropic
surrogate instead is a measurable mistake, asserted in the tests.

Quantizing *first* does not work, and the reason is structural rather than empirical: the
SVD factors of a grid-valued matrix are not themselves grid-valued, so they cost fp16, and
the storage is then **exactly** the low-rank-only cost — the first quantization bought
nothing. That is an identity, and the test asserts it with `assertEqual` and no tolerance.
Re-quantizing the factors recovers the storage and pays the rounding error twice, for
0.5% more error than factoring first.

**The refit.** Once `B` is quantized, `A` is no longer optimal — it was chosen for the
exact `B`. The least-squares correction

```
A*  =  (W X)(B̂X)ᵀ [ (B̂X)(B̂X)ᵀ ]⁻¹
```

costs one `r × r` solve. It, and activation-aware factor quantization, pay off exactly in
proportion to how much of the error is rounding rather than truncation
([`results/composition_order.csv`](results/composition_order.csv)):

| Regime | rounding share | aware + refit vs naive RTN |
|---|---|---|
| `r = 8`, truncation-limited | 3.8% | 1.020× |
| `r = 32`, balanced | 33% | 1.069× |
| `r = 252`, rounding-limited | 99.9% | 1.098× |

A first version of this study used `r = 61/125/252`, which *reads* like a spread but is
70/95/99.9% rounding — three samples of one regime, and the ablations correctly showed no
variation. The fix was to choose the ranks by measured rounding share rather than by
eye.

## Running it

No installation, no downloads, no network. numpy is the only requirement.

```bash
make test        # 49 tests, ~4 s
make results     # regenerates every CSV in results/, ~4 min
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

The suite contains the assertions the project rests on, marked in the source with
`=== THE TEST THAT MATTERS ===`:

- whitened SVD beats plain truncated SVD on `‖(W − Ŵ)X‖_F` at equal rank on
  ill-conditioned `X`, over 6 seeds, by 3.04×–3.28× (worst seed, not the mean) —
  [`tests/test_factorize.py`](tests/test_factorize.py)
- and the advantage **collapses to 1.003× on isotropic activations**, where `S` is a
  multiple of the identity and there is nothing to exploit. Two orders of magnitude
  less advantage, from identical code. That test oversamples 64× (64 channels, 4096
  samples) so the sampled `M` sits close to its isotropic population value; the
  headline experiment above oversamples only 2× (256 channels, 512 samples), and the
  1.04×–1.20× residual it reports in-sample is that difference — Wishart noise in `M`,
  which the held-out column then charges 8–13% for.
- plain truncated SVD is never beaten in the unweighted Frobenius norm, by
  Eckart–Young and by construction
- `‖E S‖² = ‖E X‖² + λ‖E‖²` holds to 9 decimal places at four ridge values — the
  identity the ridge argument is made of
- the truncation is Pythagorean in the whitened domain, which an `S`/`Sᵀ` mix-up in
  the map-back breaks without crashing anything
- an undamped rank-deficient second moment **raises**, rather than returning an `S⁻¹`
  that quietly amplifies noise — [`tests/test_whiten.py`](tests/test_whiten.py)
- greedy and Lagrangian allocation beat uniform and land within 5% of the knapsack
  optimum, and the Lagrangian solution equalizes marginal loss per parameter across
  layers — [`tests/test_allocate.py`](tests/test_allocate.py)
- **quantization wins at 4 bits and loses at 2 bits**, at matched achieved compression,
  in both directions and in the isotropic control too — the crossover of §5 asserted as
  a two-sided inequality, not a plot to squint at —
  [`tests/test_compose.py`](tests/test_compose.py)
- quantizing before factoring leaves the storage **exactly** equal to the low-rank-only
  cost, so it cannot save anything. Asserted with `assertEqual` and no tolerance,
  because it is an identity rather than a measurement

## Results

| File | Produced by | Contents |
|---|---|---|
| [`results/pareto.csv`](results/pareto.csv) | `analysis/pareto.py` | compression vs error for plain and whitened SVD, 2 regimes × 4 ridges × 7 ranks × 5 seeds, scored in-sample, held-out, and under two distribution shifts. In the isotropic block the two shift columns coincide with the held-out column, and must: an isotropic covariance has no eigenbasis to rotate, so the shift is the identity map there |
| [`results/spectra.csv`](results/spectra.csv) | `analysis/spectra.py` | full singular-value spectra of `W` and `WS` per layer, plus `cond(M)` before and after damping and stable ranks |
| [`results/allocation.csv`](results/allocation.csv) | `analysis/allocation_compare.py` | four allocators at six budgets, with proxy loss, measured squared error, achieved compression and the gap to the optimum |
| [`results/composition.csv`](results/composition.csv) | `analysis/composition.py` | the crossover: quantize-only against the best low-rank-family configuration at the same achieved compression, 4 activation spectra × 5 bit-widths × 3 seeds, with the winning `(r, b)` recorded |
| [`results/composition_order.csv`](results/composition_order.csv) | `analysis/composition.py` | ordering and ablations: low-rank→quantize vs quantize→low-rank vs quantize→low-rank→requantize, crossed with aware/RTN factor quantization and refit/no-refit, at three ranks spanning truncation- to rounding-limited |

### Why synthetic layers

The generator in [`src/synth.py`](src/synth.py) makes the anisotropy of `X` an
independent variable and controls nothing else that matters: the covariance spectrum
spans a specified condition number, a few channels can be spiked, and the weight
matrix gets its own controlled spectrum (default: one decade, leaving a 256-column
layer with stable rank ≈ 55, flat enough that plain SVD has nothing cheap to discard).
That is deliberate. The entire effect measured here exists *only* because `X` is
anisotropic — the isotropic row of the headline proves it — so the instrument has to
set that variable directly. A real model would confound it with tokenizer effects,
layer type and depth.

## The limitation I volunteer first

**At equal compression, quantization beats low-rank factorization everywhere I can
measure both — up to 5.22×.** I am saying this before I am asked, and I have run the
experiment rather than citing it: §5 above. At 3.94× compression quantization is 2.3×
more accurate than the best factored configuration on anisotropic activations. A
candidate presenting low-rank as strictly superior did not read the field.

The two useful qualifications, both measured: the crossover is at 5.22–7.76×, and past
7.76× quantization-only is not merely worse but *unreachable*, since 2 bits is the floor
of a uniform grid. And **every configuration that wins at any compression above 2× is a
composed one** — factored *and* quantized. So the honest framing is not "which method",
it is "how to split the budget", which is what §5 measures.

**Second: whitening buys accuracy with a bet on the calibration distribution, and I
measured the price.** The same 2× factorization scored on activations whose covariance
eigenbasis has been rotated at matched spectrum
([`results/pareto.csv`](results/pareto.csv)):

| Evaluation `X` | Plain SVD | Whitened SVD |
|---|---|---|
| calibration set (in-sample) | 0.5534 | 0.1689 |
| held-out, same distribution | 0.5540 | 0.2025 |
| covariance rotated half way | 0.5525 | 0.4854 |
| covariance rotated fully | 0.5553 | **0.7046** |

Plain SVD is flat across all four, because it never looked at any activations.
Whitened SVD degrades, and **somewhere between a half and a full rotation it becomes
worse than the method it beats by 2.74× in-domain.** Raising the ridge to `λ/mean(diag M) = 1` recovers part
of that (0.6044 fully-shifted) at the cost of in-domain accuracy — which is exactly
what equation (3) says a ridge does. In a real deployment this is the risk that a
calibration set drawn from one domain silently mis-serves another.

**Third: everything above is measured on synthetic layers, not a trained model.** The
mechanism is isolated and the numbers are real, but perplexity on a real transformer
is the number a team would actually want and it is not here. Forward-hook activation
capture, WikiText-2, `lm-eval`, and GGUF export are all listed in
[STATUS.md](STATUS.md) as not built, with what each needs. I would rather ship a
result I can defend completely than a headline I would have to qualify.

**Fourth: the layers here are independent, not chained.** Rank allocation is treated
as a per-layer budget problem, so `Σ_ℓ L_ℓ(r_ℓ)` ignores the fact that layer `ℓ`'s
output error becomes layer `ℓ+1`'s input perturbation. That composition is a real
effect and it is not modelled here.

## References

- [ARA: Adaptive Rank Allocation for Efficient LLM SVD
  Compression](https://arxiv.org/html/2510.19389) — the rank-allocation half of this
  repo.
- [IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM
  Compression](https://arxiv.org/html/2605.15626v1) — whitening on both sides of the
  layer, which this repo does not do; only the input side is implemented here.
- [Swift-SVD: Activation-Aware Low-Rank Compression for LLM Weights and KV
  Cache](https://www.zhongzhuzhou.org/blog/2026-05-08-2026-05-08-SwiftSVD-technical-review-en/)
- Eckart, C. & Young, G. (1936). [*The approximation of one matrix by another of lower
  rank*](https://doi.org/10.1007/BF02288367), Psychometrika 1(3):211–218 — the theorem
  whose norm is deliberately being replaced. Worth reading precisely because the
  project turns on what it does *not* say.

Full brief: [BRIEF.md](BRIEF.md).
