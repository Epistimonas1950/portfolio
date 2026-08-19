# 02 · A Quantizer Built From Scratch, With the Error Analysis

> Post-training quantization as a constrained least-squares problem, implemented from
> the mathematics rather than called from a library — with the cross-layer error bound
> derived, and then measured against what actually happens.

**Status:** the mathematics runs, is tested, and produces the numbers below on
synthetic layers with numpy alone. The real-model extension is scaffolded, not run —
see [STATUS.md](STATUS.md).

---

## The headline result

Activation-weighted relative error `‖(W − Ŵ)X‖_F / ‖WX‖_F`, mean over 5 seeds, on
synthetic layers with a deliberately anisotropic activation covariance
(condition number 10⁴) and four spiked outlier channels. Regenerate with
`make results`; raw numbers in [`results/bits_vs_error.csv`](results/bits_vs_error.csv).

| Method | eff. bits/weight | 8 bit | 4 bit | 3 bit | 2 bit |
|---|---|---|---|---|---|
| RTN, per-tensor | b + 0.125 | 0.00954 | 0.17688 | 0.40895 | 0.93787 |
| RTN, per-channel | b + 0.125 | 0.00685 | 0.12450 | 0.29456 | 0.80411 |
| Sequential, natural order | b + 0.125 | 0.00339 | 0.06255 | 0.14840 | 0.43859 |
| **Sequential, salience order** | b + 0.125 | **0.00160** | **0.02891** | **0.06824** | **0.20967** |
| Sequential + fp16 outlier channels | b + 0.33 | 0.00117 | 0.02114 | 0.04947 | 0.14747 |

**4.3× lower activation-weighted error than per-channel round-to-nearest at 3 and 4
bits**, at the same bits per weight. Error compensation contributes a factor of 2;
ordering the columns by activation salience contributes the other 2.2.

And the figure the project exists for — the cross-layer error bound against the
measured error, 3-bit, 10 layers
([`results/error_propagation.csv`](results/error_propagation.csv)):

| Layer | local error | predicted (bound) | measured | bound / measured |
|---|---|---|---|---|
| 1 | 3.4045 | 3.4045 | 3.4045 | **1.00** |
| 2 | 1.5774 | 5.6448 | 2.5412 | 2.22 |
| 4 | 0.3613 | 9.3298 | 0.9412 | 9.91 |
| 6 | 0.0799 | 12.4791 | 0.2825 | 44.2 |
| 8 | 0.0183 | 17.1832 | 0.0898 | 191 |
| 10 | 0.0041 | 22.6614 | 0.0256 | **887** |

The bound is exact at depth 1, as it must be, and then loses a factor of ≈2.1 per
layer. That decay rate is the result — see [the mathematics](#3-cross-layer-error-propagation),
below.

---

## The problem

Quantization maps each weight to a low-bit grid. Round-to-nearest is the obvious
method, and it is **exactly optimal** — for

```
minimize   ‖W − Ŵ‖²_F      over Ŵ on the grid
```

which decouples into independent per-weight roundings. But nothing downstream cares
about `‖W − Ŵ‖_F`. What a layer is for is producing `WX`, so the objective that
matters is

```
minimize   ‖WX − ŴX‖²_F     over Ŵ on the grid
```

a constrained least-squares problem over a discrete set. RTN is blind to `X`. Once
the objective is written this way, the entire GPTQ line of work stops being a trick
and becomes the obvious relaxation.

**Scoped at ≤4 bits deliberately.** At 8 bits the grid is fine enough that the method
hardly matters — the table above shows RTN already at 0.7% relative error there. An
8-bit result is not a result, and this repo says so rather than presenting one.

## The mathematics

### 1. The Hessian is the activation second moment

Expanding the objective with `E = W − Ŵ` gives `L = tr(E X Xᵀ Eᵀ)`, quadratic in each
row of `E` independently, with the same Hessian for every row:

```
H = 2 X Xᵀ
```

One factorization serves the whole layer. It is also, up to the factor 2, the matrix
the low-rank project ([`01`](../01-lowrank-llm)) whitens by — two different compression
methods, one statistic. → [`src/hessian.py`](src/hessian.py)

### 2. Error compensation

Decide columns one at a time, and after each rounding, correct the columns not yet
decided. Rounding column `q` incurs `δ = w_q − ŵ_q`; the optimal-brain-surgeon update
that best restores the layer output is

```
δ_remaining  =  −( δ / [H⁻¹]_qq ) · H⁻¹[:, q]
```

restricted to the undecided coordinates. In terms of the upper Cholesky factor `R` of
`H⁻¹` (so `RᵀR = H⁻¹`) this is a rank-one update of the tail:

```
W[:, q+1:]  −=  (δ / R_qq) ⊗ R[q, q+1:]
```

`H⁻¹` is never formed. One `O(n³)` factorization replaces an explicit inverse per
column. → [`src/sequential.py`](src/sequential.py)

**Damping.** `H` is singular in practice — a calibration set that does not excite
every input channel produces a rank-deficient `X Xᵀ`, and with 192 samples for 256
channels it is singular by construction. Add `λ · mean(diag H) · I`. Scaling by the
mean diagonal is what makes `λ` transferable between layers whose activation energies
differ by orders of magnitude. The sweep
([`results/damping_sweep.csv`](results/damping_sweep.csv)) shows both failure modes:

| λ ratio | Cholesky failures | improvement over RTN |
|---|---|---|
| 0 | **5 / 5** | — |
| 1e-6 | 0 | 3.66× |
| **1e-3** | 0 | **3.79×** |
| 1e-1 | 0 | 3.25× |
| 1 | 0 | 2.17× |
| 10 | 0 | 1.31× |

Too little and it does not factor; too much and `H` is pushed toward a multiple of
the identity, the compensation term toward zero, and the method degrades continuously
back into round-to-nearest. The last column heading for 1.0 is the proof of that
statement.

**Ordering.** Whichever column goes last has nothing left to absorb its error.
Quantizing by descending `diag(H)` — the most-excited input channels first — spreads
the largest errors over the largest remaining set, and is worth a further 2.2× over
natural order. A magnitude ordering that ignores activations is included as the
control. → [`src/ordering.py`](src/ordering.py)

### 3. Cross-layer error propagation

Layer `ℓ`'s output error is layer `ℓ+1`'s *input* perturbation, so per-layer figures
do not compose by addition. With `E_ℓ = a_ℓ − â_ℓ`:

```
E_ℓ = (W_ℓ − Ŵ_ℓ) a_{ℓ−1}  +  Ŵ_ℓ E_{ℓ−1}
```

and by the triangle and submultiplicative inequalities,

```
‖E_ℓ‖_F  ≤  ‖(W_ℓ − Ŵ_ℓ) a_{ℓ−1}‖_F  +  ‖Ŵ_ℓ‖₂ ‖E_{ℓ−1}‖_F
```

which unrolls to a sum of local errors weighted by downstream operator norms. The
local term is exactly the per-layer objective, evaluated on clean activations — so it
is free at quantization time.

**Where it is loose, and why that is the finding.** The bound is tight at depth 1 and
loses ≈2.1× per layer thereafter. Both inequalities assume a worst case that does not
occur: the triangle inequality assumes the fresh rounding error points the same way as
the error arriving from below, and the operator norm assumes that error lands on the
top singular direction of the next layer. Independent rounding decisions produce
errors that are close to orthogonal and land isotropically, so each layer contracts
them by something near its *typical* singular value rather than its largest. The
factor of 2.1 per layer is the measured ratio between those two quantities. A tight
bound here would mean the layers' rounding errors were conspiring — which would itself
be worth reporting. → [`src/propagation.py`](src/propagation.py),
[`analysis/error_bound.py`](analysis/error_bound.py)

### 4. Outlier channels

A few input channels carry activations an order of magnitude above the rest, so they
dominate `‖(W − Ŵ)X‖_F` while occupying the same share of the grid as everything else.
Detect them from `diag(H)` and keep those columns in fp16: 4 columns out of 256 costs
0.20 bits per weight and buys a further 1.4×. The cost is accounted for in
`grid.effective_bits`, which counts scales and zero-points too — a "4-bit" quantizer
with per-32-column group scales is really carrying 5 bits per weight, and comparing it
against a per-tensor baseline as though it were not is the standard way these tables
mislead. → [`src/outliers.py`](src/outliers.py), [`src/grid.py`](src/grid.py)

## Running it

No installation, no downloads, no network. numpy is the only requirement.

```bash
make test        # 20 tests, ~1 s
make results     # regenerates every CSV in results/
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

The suite includes the assertions the whole project rests on, marked in the source
with `=== THE TEST THAT MATTERS ===`:

- error compensation beats RTN on `‖(W − Ŵ)X‖_F` at 3 bits, over 5 seeds, by more
  than 2× — [`tests/test_sequential.py`](tests/test_sequential.py)
- and the advantage **collapses on isotropic activations**, where `H` is a multiple of
  the identity and there is nothing to exploit. If it did not, the improvement would be
  coming from somewhere other than the mathematics advertised here.
- RTN is never beaten in weight space, by construction
- the propagation bound dominates the measurement at every depth —
  [`tests/test_numerics.py`](tests/test_numerics.py)
- the undamped singular Hessian raises rather than returning quietly wrong numbers

## Why synthetic layers

The generator in [`src/synth.py`](src/synth.py) controls the two properties that make
real transformer layers hard — a spectrum spanning several decades, and a handful of
spiked channels — and nothing else. That is deliberate. On isotropic activations the
activation-weighted objective and the weight-space objective coincide and RTN is
optimal; the entire effect being measured here exists only because `X` is anisotropic,
so the instrument has to control that variable directly. A real model would confound
it with tokenizer effects, layer-type effects and depth.

The real-model run is the obvious next step and is scaffolded, not faked. See
[STATUS.md](STATUS.md).

## The limitation I volunteer first

**I am re-implementing a known method.** GPTQ is published, its reference
implementation is public, and I am not claiming otherwise. The claim is narrower and
more useful: I implemented it from the objective, I can derive the compensation update
at a whiteboard, and I went looking for where its numerical assumptions break — which
is what the damping sweep and the propagation bound are.

**Second: everything above is measured on synthetic layers, not a real model.** The
mechanism is isolated and the numbers are real, but perplexity on a trained transformer
is the number a team would actually want and it is not here yet. I would rather ship a
result I can defend completely than a headline I would have to qualify.

**Third: round-to-nearest is genuinely fine at 8 bits.** The 8-bit column is in the
table to make that point, not to claim a win there.

## References

- Frantar, E. et al. [*GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers*](https://arxiv.org/abs/2210.17323) (arXiv:2210.17323) —
  the method reproduced here. Reference code:
  [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq).
- Hassibi, B. & Stork, D. [*Second Order Derivatives for Network Pruning: Optimal Brain
  Surgeon*](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
  (NeurIPS 1992) — where the `H⁻¹` compensation update comes from.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, SIAM — book, no free
  copy. The authority for the Cholesky and error-propagation arguments.

Full brief: [BRIEF.md](BRIEF.md).
