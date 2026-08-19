# 01 · Activation-Aware Low-Rank Compression of a Small LLM

> Make a 0.5–1.5B model materially smaller using SVD done *correctly*, then run it on a
> Raspberry Pi 3.

| | |
|---|---|
| **Effort** | 3–4 weeks |
| **Prerequisites** | None — **build this first** |
| **Feeds** | `02` (quantization extends this codebase), `07` (Pi 3 deployment) |
| **Math** | SVD, Eckart–Young, Cholesky, constrained optimization, conditioning |
| **Status** | ☐ not started |

---

## The problem

You want a smaller model. The obvious move is to factor every weight matrix `W ≈ AB` with
`rank(A) = rank(B) = r ≪ min(m, n)`, replacing `mn` parameters with `r(m + n)`.

The obvious *method* is truncated SVD, justified by Eckart–Young: the rank-`r` truncation
minimizes `‖W − Ŵ‖_F` over all rank-`r` matrices.

**This is the trap, and it is the whole project.** Eckart–Young optimality is in the *unweighted*
Frobenius norm. Nobody cares about `‖W − Ŵ‖_F`. What changes the model's output is the
activation-weighted error

```
‖(W − Ŵ) X‖_F
```

where `X` holds the activations that layer actually sees. Truncating the smallest singular values
of `W` throws away directions that may be heavily excited by real input, while keeping directions
the model never visits. Naive SVD compression of LLMs collapses for exactly this reason.

## The mathematics

**1. Whitening.** Collect activations `X` for a layer over a calibration set and form the second
moment `M = XXᵀ`. Take a Cholesky factor `M = SSᵀ`. Then

```
‖(W − Ŵ)X‖_F  =  ‖(W − Ŵ)S‖_F
```

so decompose `WS = UΣVᵀ`, truncate *that* to rank `r`, and map back: `Ŵ = U_r Σ_r V_rᵀ S⁻¹`.
Eckart–Young now applies in the norm you actually care about. This one transform is the
difference between the method working and not working.

**2. Conditioning.** `M` is routinely near-singular — some activation directions are barely
excited, and `S⁻¹` amplifies them. Add ridge regularization `M + λI`, and justify your `λ`.
Report the condition number. This is a place to show numerical judgment rather than hope.

**3. Rank allocation.** Uniform compression ratio across layers is leaving performance on the
table. Given a global parameter budget `B`, solve

```
minimize   Σ_ℓ  L_ℓ(r_ℓ)      subject to   Σ_ℓ  r_ℓ (m_ℓ + n_ℓ)  ≤  B
```

where `L_ℓ(r)` is the estimated truncation loss for layer `ℓ` at rank `r` — available almost free
from the tail of the singular-value spectrum, `Σ_{i>r} σ_i²`. Solve it as a Lagrangian
relaxation (equalize marginal loss per parameter across layers), a greedy marginal-gain sweep, or
an explicit knapsack. Compare all three; they don't agree, and explaining why is a good interview
moment.

## What to build

- [ ] Calibration-set activation capture with forward hooks (WikiText-2 or C4 subset)
- [ ] Whitened SVD factorization per linear layer, with ridge damping and condition reporting
- [ ] Three rank-allocation strategies: uniform, greedy marginal-gain, Lagrangian
- [ ] Reconstruction into a runnable model (fold `A`, `B` into two `nn.Linear` layers)
- [ ] Evaluation harness: perplexity + at least one downstream task
- [ ] Sweep script producing the compression-vs-quality Pareto curve
- [ ] Export path to GGUF for `llama.cpp` (this is what project `07` deploys)

**Model choice:** SmolLM2-360M for fast iteration; Qwen2.5-0.5B or Llama-3.2-1B for the headline
numbers.

## How it's measured

The deliverable is a curve, not a single number. Every point needs a baseline at equal
compression.

| Method | Compression | Perplexity ↓ | Downstream acc ↑ | Pi 3 tok/s | Peak RSS |
|---|---|---|---|---|---|
| Uncompressed baseline | 1.00× | | | | |
| Plain truncated SVD | | | | | |
| Whitened SVD, uniform rank | | | | | |
| Whitened SVD, greedy alloc | | | | | |
| Whitened SVD, Lagrangian alloc | | | | | |

Also plot: singular-value spectra before and after whitening (they look very different — this is
your best single figure), and condition number per layer.

## The limitation you volunteer first

**At equal compression, 4-bit quantization usually beats low-rank factorization on modern LLMs.**
Say this before you are asked. The interesting results are (a) *where* the crossover sits, and
(b) that the two methods **compose** — which is what project `02` measures. A candidate who
presents low-rank as strictly superior looks like they didn't read the field; a candidate who
maps the crossover looks like they ran the experiment.

Secondary: whitening needs a calibration set, so you inherit a distribution-shift risk. Measure
perplexity on a domain the calibration set didn't cover and report the gap.

## Interview claim

> I cut the model by X% with Y perplexity degradation, versus Z for naive truncated SVD at the
> same ratio — and I can tell you exactly which norm the naive method is optimal in and why that
> norm is the wrong one.

## Stack

PyTorch · NumPy/SciPy (`scipy.linalg.cholesky`, `svd`) · `lm-eval-harness` · `llama.cpp` for the
edge deploy

## Suggested repo layout

```
lowrank-llm/
  README.md              <- the curve goes at the top, above the fold
  src/
    calibrate.py         activation capture
    factorize.py         whitened SVD + damping
    allocate.py          uniform | greedy | lagrangian
    rebuild.py           factored model reconstruction
    export_gguf.py
  eval/
    perplexity.py
    sweep.py
  results/
    pareto.png           <- the figure everything else supports
    spectra.png
    results.csv
```

## References

- [ARA: Adaptive Rank Allocation for Efficient LLM SVD Compression](https://arxiv.org/html/2510.19389)
- [IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM Compression](https://arxiv.org/html/2605.15626v1)
- [Swift-SVD: Activation-Aware Low-Rank Compression for LLM Weights and KV Cache](https://www.zhongzhuzhou.org/blog/2026-05-08-2026-05-08-SwiftSVD-technical-review-en/)
- Eckart, C. & Young, G. (1936). [*The approximation of one matrix by another of lower
  rank*](https://doi.org/10.1007/BF02288367), Psychometrika 1(3):211–218 — the theorem whose norm
  you are deliberately replacing. Worth reading precisely because the project turns on what it
  does *not* say.
