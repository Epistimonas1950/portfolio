# 02 · A Quantizer Built From Scratch, With the Error Analysis

> INT8 and INT4 post-training quantization as a numerical-analysis problem, not a library call.

| | |
|---|---|
| **Effort** | 2 weeks |
| **Prerequisites** | `01` — reuses its calibration and evaluation harness |
| **Feeds** | `07` (Pi 3 deployment), and the composition experiment with `01` |
| **Math** | Least squares, Hessians, Cholesky ordering and damping, error propagation |
| **Status** | ☐ not started |

---

## The problem

Quantization maps each weight to a low-bit grid. Round-to-nearest (RTN) is the obvious method and
at 8 bits it is nearly free — there is almost nothing to win there. **Scope this project at ≤4
bits from the start**, or your result reads as a rediscovery of something everyone already knows.

Below 4 bits the naive approach falls apart, and the reason is mathematical: rounding each weight
independently minimizes weight-space error, but weight-space error is not what degrades the
model. The layer's job is to produce `WX`. So the objective is

```
minimize   ‖WX − ŴX‖²_F      over  Ŵ  on the quantization grid
```

which is a constrained least-squares problem over a discrete set — and once you write it that
way, the entire GPTQ line of work becomes obvious rather than magic.

## The mathematics

**1. The Hessian is the activation second moment.** Differentiating the objective twice gives

```
H = 2 XXᵀ
```

the same matrix that project `01` whitens by. Two different compression methods, one statistic —
worth pointing out in an interview.

**2. Error compensation.** Quantize weights one at a time. When you round weight `w_q` and incur
error `δ`, the optimal-brain-surgeon update redistributes that error into the *not-yet-quantized*
weights so the layer output is corrected:

```
δ_remaining  =  − (δ / [H⁻¹]_qq) · H⁻¹_{:,q}
```

Each rounding decision is therefore compensated by the weights that haven't been decided yet.
This is why sequential quantization crushes independent rounding at low bit-width.

**3. The numerics are the interesting part.** `H⁻¹` is never formed explicitly — you work through
a Cholesky factor, and:

- **Damping.** `H` is singular in practice (dead activation directions). Add `λ · mean(diag H)`;
  too little and the factorization fails, too much and the compensation becomes useless. Sweep it
  and plot.
- **Ordering.** The order in which you quantize columns changes the result, because each step's
  compensation depends on what remains. Compare natural order against magnitude-descending and
  activation-salience order.
- **Error propagation across layers.** Layer `ℓ`'s output error becomes layer `ℓ+1`'s input
  perturbation. Derive a bound — even a loose product-of-operator-norms bound — then measure the
  actual propagated error and plot predicted vs. measured. **This plot is the project.** It is
  the single most mathematician-flavoured artifact in the whole portfolio.

**4. Outliers.** A handful of activation channels have magnitudes orders larger than the rest, and
they dominate the error. Per-channel scales, or keeping those channels in higher precision, is the
standard fix. Show the magnitude histogram; show what happens when you don't handle them.

## What to build

- [ ] Grid utilities: symmetric/asymmetric, per-tensor vs. per-channel scale and zero-point
- [ ] Baseline: round-to-nearest at 8/4/3 bits
- [ ] Sequential error-compensating solver via Cholesky, with configurable damping and ordering
- [ ] Outlier-channel detection and mixed-precision handling
- [ ] Cross-layer error-propagation bound: derivation in the README, measurement in code
- [ ] Composition experiment: `01`'s low-rank output, then quantized — does the ordering matter?
- [ ] GGUF export so `07` can run it

## How it's measured

| Method | Bits/weight | Perplexity ↓ | Δ vs. FP16 | Wall clock |
|---|---|---|---|---|
| FP16 baseline | 16 | | — | |
| RTN per-tensor | 4 | | | |
| RTN per-channel | 4 | | | |
| Yours, natural order | 4 | | | |
| Yours, salience order | 4 | | | |
| Reference (GPTQ / bitsandbytes) | 4 | | | |
| Yours at 3 bits | 3 | | | |
| **`01` low-rank → yours** | ~4 eff. | | | |

Plus: predicted vs. measured error propagation, and the damping sweep.

## The limitation you volunteer first

**Round-to-nearest is already very strong at 8 bits.** If you present an 8-bit result as an
achievement you look naive. Lead with the ≤4-bit regime and say explicitly why 8-bit is not
interesting.

Second: you are re-implementing a known method. That is fine — say so plainly. The claim is not
"I invented this," it is "I implemented it from the mathematics, I can derive the update rule at
the whiteboard, and I found where its numerical assumptions break."

## Interview claim

> I understand why 8-bit is nearly free and 4-bit is not, where the error actually comes from, and
> I have the outlier-channel data to prove it.

## Stack

PyTorch · NumPy/SciPy · `bitsandbytes` or `GPTQ` as the reference to match · `llama.cpp` for
deployment

## Suggested repo layout

```
scratch-quantizer/
  README.md              <- predicted-vs-measured error plot at the top
  src/
    grid.py              scales, zero-points, symmetric/asymmetric
    rtn.py               baseline
    sequential.py        Cholesky + error compensation
    ordering.py          column ordering strategies
    outliers.py
  analysis/
    error_bound.py       the derivation, executable
    damping_sweep.py
  results/
    predicted_vs_measured.png
    bits_vs_ppl.png
```

## References

- Frantar, E. et al. [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers*](https://arxiv.org/abs/2210.17323) (arXiv:2210.17323) — the method you reproduce.
  Reference code: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq).
- Hassibi, B. & Stork, D. [*Second Order Derivatives for Network Pruning: Optimal Brain
  Surgeon*](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
  (NeurIPS 1992) — where the `H⁻¹` error-compensation update comes from.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, SIAM — book, no free copy. The
  authority for the Cholesky and error-propagation arguments you'll want to cite properly.
