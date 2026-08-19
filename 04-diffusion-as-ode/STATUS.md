# Status

Honest accounting of what runs, what is stubbed, and what is missing. The portfolio's
whole thesis is that measurements are real, so this file is part of the deliverable.

**Last verified:** `make test` → 54 tests, OK, ~7 s. `make results` → 5 CSVs
regenerated, ~35 s. Python 3.12.3, numpy 1.26.4, no other dependency, no network.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| VP forward SDE: β(t), α(t), σ(t), drift/diffusion, log-SNR and its inverse | `src/sde.py` | complete, tested |
| VE forward SDE (secondary; used to show α ≡ 1 empties the linear part) | `src/sde.py` | complete, tested |
| Gaussian mixture: exact score, CDF, quantile, moments, sampling | `src/sde.py` | complete, tested |
| Exact analytic solution of the prob.-flow ODE (affine, single Gaussian) | `src/reference.py` | complete, tested |
| Exact analytic solution via the 1-D quantile transport (any mixture) | `src/reference.py` | complete, tested |
| Very-high-NFE numerical reference (DPM-Solver-2, 4096 steps) | `src/reference.py` | complete, tested |
| Time grids: uniform in `t`, uniform in log-SNR | `src/schedule.py` | complete, tested |
| NFE counter and result record | `src/nfe.py` | complete, tested |
| Euler–Maruyama on the reverse SDE, with common-random-number support | `src/samplers/euler_maruyama.py` | complete, tested |
| Euler on the probability-flow ODE | `src/samplers/euler_ode.py` | complete, tested |
| Heun (explicit trapezoidal) | `src/samplers/heun.py` | complete, tested |
| Exponential integrator, DPM-Solver orders 1 and 2 | `src/samplers/exponential.py` | complete, tested |
| Adaptive Heun/Euler pair with PI step-size control | `src/samplers/adaptive.py` | complete, tested |
| W₁, energy distance, moments, mode-weight TV — all exact against the target | `src/metrics.py` | complete, tested |
| Test problem, prior choices, flow-map conditioning | `src/problem.py` | complete |
| Convergence-order study (ODE trajectory, SDE strong, SDE weak) | `analysis/convergence_order.py` | runs, CSV committed |
| NFE-vs-quality frontier and the target table | `analysis/nfe_vs_quality.py` | runs, 2 CSVs committed |
| Stochastic/deterministic tradeoff, conditional diversity, invertibility | `analysis/stochastic_vs_deterministic.py` | runs, CSV committed |
| Stiffness, signed amplification factors, prior-sharpness crossover | `analysis/stability.py` | runs, CSV committed |
| Column contract of the committed CSVs, checked without matplotlib | `tests/test_outputs.py` | complete |

Committed outputs: `results/convergence.csv`, `results/nfe_quality.csv`,
`results/nfe_targets.csv`, `results/sde_vs_ode.csv`, `results/stability.csv`.

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed on this machine. It exits with a message naming the package rather than failing obscurely. Every number it would draw is already in `results/*.csv`, which are committed. The brief's headline deliverables are `.png`s; the numbers behind them are all here. Because the file cannot be executed, `tests/test_outputs.py` asserts every CSV column it indexes by name actually exists and that every filter it applies selects a non-empty subset — so a typo in it fails the suite rather than waiting for someone with matplotlib. |

## Not built

These need something this machine does not have, and they are **absent rather than
mocked**. No number in this repo is a placeholder or an estimate.

- **A real pretrained diffusion model.** No torch, no `diffusers`, no GPU, no network.
  Everything here uses an analytic score. This is a deliberate design choice as well as
  a constraint — it is what makes the convergence orders measurable against an exact
  solution — but it means no claim in this repo is about a trained network's behaviour.
- **Real images, and FID.** FID needs an Inception network and a dataset. The brief's
  headline column is *NFE for FID ≤ target*; I substituted 1-Wasserstein against the
  known target law and said so in the README rather than reusing the FID heading for a
  different metric. There is no FID number anywhere in this repo, and there should not
  be one until it is computed.
- **Anything above one dimension.** The samplers, schedule, score and the affine
  Gaussian reference are all dimension-agnostic and would run on `(n, d)` arrays
  unchanged, but the exact quantile transport used as ground truth is a 1-D argument,
  and every measurement here is 1-D. A 2-D study with the high-NFE reference standing in
  for the exact one is the obvious next step and is not done.
- **A DPM-Solver-3 / multistep (Adams-style) member.** Orders 1 and 2 are implemented;
  the third-order single-step and the multistep variants of Lu et al. are not. The
  brief's table lists the exponential integrator at "2–3"; what is measured here is 2.
- **Part B of the brief — the classical-vs-neural PDE Pareto front** (adaptive RK45,
  implicit BDF, a spectral method, a PINN, a Fourier neural operator on one stiff
  problem). The brief marks it optional; the PINN and FNO need torch, which is not
  installed. Nothing in this repo pretends to have done it.

## What the numbers here do and do not support

**Supported.** The integrators are correct: their measured orders are 0.993, 2.017,
0.996 and 2.015 against an exact solution, with log–log fit residuals below 0.005
decades over a stated window. Euler–Maruyama's strong order on this additive-noise SDE
is 1.001, not the ½ usually quoted. The exponential integrator is exact in one step on a
point-mass prior and its deficit for `v > 0` matches a closed form to 11 places. Second-
order methods reach a fixed trajectory accuracy in 8× fewer network evaluations than
first-order Euler, and PI-controlled adaptive stepping in 10×, against a uniform-log-SNR
baseline. The exponential integrator's advantage over Euler crosses unity at prior
variance ≈ 0.01. On this schedule explicit Euler is never outside its stability region
on a monotone grid (the measured `min(1 + h·∂F/∂x)` bottoms out at +0.18 over both
priors, both grid families, 4 to 128 steps); what fails on a sharply multimodal prior at
8 steps is accuracy, an expanding step of `e^{4.66} = 106` reproduced as `5.66`.

**Not supported.** Any claim about a trained score network, about image quality, about
FID, about wall-clock time on a GPU, or about how much of the 8–10× NFE reduction
survives contact with a real model. The published figure for that is 2–4× (DPM-Solver,
EDM); I have not measured it and do not claim it.
