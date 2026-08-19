# Status

Honest accounting of what runs, what is stubbed, and what is missing. The portfolio's
whole thesis is that the measurements are real, so this file is part of the
deliverable.

**Last verified:** `make test` → 49 tests, OK, 4.2 s. `make results` → 5 CSVs
regenerated, 4 m 07 s, byte-identical to the committed ones. Python 3.12.3, numpy 1.26.4, no other dependency, no network.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| Second moment `M = XXᵀ`, ridge, Cholesky, conditioning report | `src/whiten.py` | complete, tested |
| Back-substitution map-back `B = V_rᵀS⁻¹` (no `S⁻¹` formed) | `src/whiten.py` | complete, tested against `np.linalg.solve` |
| Plain truncated SVD baseline | `src/factorize.py` | complete, tested |
| Whitened SVD with ridge damping and diagnostics | `src/factorize.py` | complete, tested |
| Parameter accounting, break-even rank, compression ratio | `src/rebuild.py` | complete, tested |
| Uniform rank allocation | `src/allocate.py` | complete, tested |
| Greedy marginal-gain allocation | `src/allocate.py` | complete, tested |
| Lagrangian-relaxation allocation | `src/allocate.py` | complete, tested |
| Exact knapsack DP over a discretised rank grid | `src/allocate.py` | complete, tested |
| Synthetic layer / stack generator, distribution shift | `src/synth.py` | complete |
| Uniform grid + sequential Cholesky-compensated quantizer | `src/quantize.py` | complete, tested — **vendored** from `02-scratch-quantizer` so this repo has no cross-repo dependency; the full treatment (damping sweep, ordering study, cross-layer bound) lives there |
| Composition: storage accounting, both orderings, factor refit | `src/compose.py` | complete, tested |
| Low-rank / quantization crossover sweep | `analysis/composition.py` | runs, CSV committed |
| Compression-vs-error sweep | `analysis/pareto.py` | runs, CSV committed |
| Spectra and per-layer conditioning | `analysis/spectra.py` | runs, CSV committed |
| Allocation comparison at six budgets | `analysis/allocation_compare.py` | runs, CSV committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed on this machine (the interpreter is PEP-668 externally managed, so `pip install` fails). It exits with a message naming the package and the venv command rather than failing obscurely. Every number it would draw is already in `results/*.csv`. |

## Not built yet

These are the parts of [BRIEF.md](BRIEF.md) that need a real model or a real dataset.
They are **absent rather than mocked** — there is no stub in `src/` pretending to
capture activations, because a stub that returns plausible arrays is worse than no
file at all.

- **Calibration-set activation capture.** Forward hooks on a small transformer
  (SmolLM2-360M / Qwen2.5-0.5B), WikiText-2 or a C4 subset, per-layer `X` accumulated
  as a running `XXᵀ` so nothing large is held. Needs torch + transformers + datasets
  (see `requirements.txt`) and a model download.
- **Reconstruction into a runnable model.** Folding `A` and `B` into two `nn.Linear`
  layers and running a forward pass. The mathematics of the fold is done and tested
  (`src/rebuild.py`); wiring it into a module tree is not.
- **Evaluation harness.** Perplexity on WikiText-2 plus one downstream task via
  `lm-eval`. Every quality number in this repo is a matrix norm, not a perplexity, and
  the README says so wherever it matters.
- **Distribution-shift measurement on real text.** The synthetic version is built and
  reported (`rel_error_shift50` / `rel_error_shift100` in `results/pareto.csv`); the
  real version is calibrating on one corpus and evaluating on another.
- **GGUF export** for `llama.cpp`, which the Raspberry Pi deployment project would
  consume.
- **Composition with quantization on a real model.** The synthetic version is **built
  and measured** (§5 of the README, `src/compose.py`, `results/composition*.csv`): the
  crossover, both orderings, and the factor-refit ablation. What is not here is the same
  sweep on trained weights, where the activation spectrum is whatever the model produces
  rather than whatever the generator was told to produce. The crossover held across four
  synthetic spectra, which is evidence that it is not an artefact of one generator — it
  is not evidence about any particular model.
- **Per-layer chained error propagation.** Allocation here treats layers as
  independent. Layer `ℓ`'s output error is layer `ℓ+1`'s input perturbation, and the
  allocation objective does not model that.

## What the numbers here do and do not support

**Supported.** The mechanism, on synthetic layers, with the anisotropy of `X` as a
controlled variable:

- whitened SVD beats plain truncated SVD by 1.15×–5.68× on held-out
  activation-weighted error at equal rank, when `cond(XXᵀ) ≈ 2.8·10⁵`;
- the advantage collapses to 1.003× on isotropic activations and *inverts* out of
  sample there, which is what says the gain comes from the advertised mathematics;
- `‖ES‖² = ‖EX‖² + λ‖E‖²`, so the ridge is a dial between the two methods, and the
  best `λ` moves with the size of the distribution shift;
- uniform rank allocation costs 23%–63% excess loss against the exact optimum, while
  greedy and Lagrangian land within 0.8%, for a reason (separable convex allocation)
  rather than by luck;
- the allocation objective `Σ_{i>r} σ²_i` equals the measured squared
  activation-weighted error to within [1.000008, 1.000057] across all 24 allocations,
  the residual being exactly the `λ‖E‖²` term at `ridge = 10⁻⁶`;
- at matched *achieved* compression, quantization beats the best low-rank-family
  configuration at 1.98×, 2.64×, 3.94× and 5.22×, and loses at 7.76× — in all four
  activation spectra tested, so the crossover bracket is not an artefact of one
  generator setting;
- every configuration that wins above 2× compression is a **composed** one (factored
  *and* quantized); fp16 factors never win;
- quantizing before factoring leaves storage exactly equal to the low-rank-only cost —
  an identity, not a measurement.

**Not supported.** Any claim about perplexity, about a specific model, or about
wall-clock speed or memory on real hardware. In particular the crossover bracket is a
statement about these synthetic spectra: it is robust across the four tested, which
makes it more than one data point, but a trained transformer's activation spectrum is
not something I chose and the number would have to be re-measured there. Those need the
work listed above.
