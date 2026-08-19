# Status

Honest accounting of what runs, what is stubbed, and what is missing. The portfolio's
whole thesis is that measurements are real, so this file is part of the deliverable.

**Last verified:** `make test` → 20 tests, OK. `make results` → 3 CSVs regenerated.
Python 3.12.3, numpy 1.26.4, no other dependency.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| Quantization grids, per-tensor / per-channel, sym / asym | `src/grid.py` | complete, tested |
| Honest bits-per-weight accounting incl. metadata | `src/grid.py` | complete, tested |
| Round-to-nearest baseline | `src/rtn.py` | complete, tested |
| Layer Hessian, damping, inverse Cholesky | `src/hessian.py` | complete, tested |
| Sequential quantization with error compensation | `src/sequential.py` | complete, tested |
| Column orderings (natural / salience / magnitude) | `src/ordering.py` | complete, tested |
| Outlier-channel detection + fp16 retention | `src/outliers.py` | complete, tested |
| Cross-layer error bound + measurement | `src/propagation.py` | complete, tested |
| Synthetic layer / stack generator | `src/synth.py` | complete |
| Bits-vs-error sweep | `analysis/bits_vs_error.py` | runs, CSV committed |
| Damping sweep | `analysis/damping_sweep.py` | runs, CSV committed |
| Propagation study | `analysis/error_bound.py` | runs, CSV committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed on this machine. Exits with a message naming the package rather than failing obscurely. Every number it draws is already in `results/*.csv`. |

## Not built yet

These are the parts of [BRIEF.md](BRIEF.md) that need a real model, and they are
absent rather than mocked:

- **Real-model calibration and evaluation.** Activation capture via forward hooks on a
  small transformer, WikiText-2 calibration, perplexity and one downstream task. Needs
  torch + transformers + datasets (see `requirements.txt`) and a model download.
- **Reference comparison against GPTQ / bitsandbytes.** The row in the brief's results
  table that says "reference" is empty for a reason.
- **The composition experiment, on a real model.** The synthetic version is **built**,
  and it lives in the sibling repo rather than here because it needs `01`'s
  factorization: see `01-lowrank-llm/src/compose.py` and its `results/composition*.csv`.
  It vendors a minimal copy of this repo's grid and sequential solver so that repo stays
  standalone. What is *not* built is the same study on trained weights, and the
  perplexity column of the results table below stays empty for that reason — the
  composition study measures a matrix norm, not a perplexity.
- **GGUF export** so project `07` can deploy the result.

## What the numbers here do and do not support

Supported: the mechanism. Error compensation reduces activation-weighted error by
~4.3× over per-channel RTN at 3–4 bits on anisotropic activations, the advantage
vanishes on isotropic ones, the damping optimum sits near `λ/mean(diag H) = 10⁻³`, and
the operator-norm propagation bound loses ≈2.1× per layer.

Not supported: any claim about perplexity, any claim about a specific model, any
claim about wall-clock speed on real hardware. Those need the work listed above.
