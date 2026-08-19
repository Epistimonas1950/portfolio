# 04 · Diffusion Samplers as Numerical Integrators

> The project that makes a numerical analyst obviously valuable to a generative-AI team.
> **This is your differential-equations work**, placed where an AI team will pay for it.

| | |
|---|---|
| **Effort** | 3 weeks (Part A) · +2 weeks (Part B, optional) |
| **Prerequisites** | None |
| **Feeds** | Nothing — this one stands alone as the differentiated story |
| **Math** | SDE/ODE integrators, convergence order, truncation error, stability, stiffness |
| **Status** | ☐ not started |

---

## Why this is the DE project, not a detour

You asked for numerical solution of differential equations. The instinct is to write a PDE solver
and show it works. But a solver that reproduces a known solution is a *verification exercise* —
correct, and worth nothing in an interview, because there is no baseline and no decision resting
on the result.

Here is the reframe: **sampling from a diffusion model is solving a differential equation.** The
reverse-time process is an SDE

```
dx  =  [ f(x,t) − g(t)² ∇ₓ log p_t(x) ] dt  +  g(t) dW_bar
```

and its deterministic counterpart, the probability-flow ODE, has the same marginals:

```
dx/dt  =  f(x,t) − ½ g(t)² ∇ₓ log p_t(x)
```

The neural network *is* the vector field (it estimates the score `∇ₓ log p_t`). "Number of
sampling steps" is *step count*. "Sampler" is *integration scheme*. Every image model in
production is running a hand-tuned ODE solver, and the people running them are usually not
numerical analysts.

That is your opening.

---

## Part A — the differentiator (build this first)

**What it is.** Take a small pretrained diffusion model. Throw away its sampler and write your
own. Measure sample quality against the number of network evaluations (NFE) — the only cost that
matters, since each step is one forward pass.

**Integrators to implement:**

| Scheme | Type | Expected order | Notes |
|---|---|---|---|
| Euler–Maruyama | SDE | 0.5 strong / 1.0 weak | the naive baseline |
| Euler | ODE (prob. flow) | 1 | deterministic baseline |
| Heun (2nd-order) | ODE | 2 | one extra eval per step |
| Exponential integrator (DPM-Solver style) | ODE | 2–3 | exploits the semi-linear structure |
| Adaptive step control | ODE | — | embedded error estimate |

**The mathematics you own:**

- **Empirical order of convergence.** Halve the step size, measure error against a
  reference solution computed with many steps, plot on log–log, read the slope. If Heun gives
  slope ≈ 2 you have proved your implementation correct — and this plot is the credibility
  anchor for the whole project.
- **Local vs. global truncation error** and why order `p` locally gives order `p` globally here.
- **Why the exponential integrator wins.** The probability-flow ODE is semi-linear: a stiff linear
  part with a known closed-form solution, plus a smooth nonlinear remainder. Integrating the
  linear part exactly and only approximating the remainder is the whole trick, and it is a
  classical numerical-analysis idea, not a deep-learning one.
- **Stochastic vs. deterministic.** The SDE sampler injects noise (higher sample diversity,
  slower convergence in NFE); the ODE is deterministic and converges faster but can lose
  diversity. This is a genuine bias–variance tradeoff with a measurable frontier.

**Headline metric.** NFE required to reach a target sample quality, per sampler. **A 2–4×
reduction in NFE at matched quality is a result an engineering team can use on Monday.**

| Sampler | NFE for FID ≤ target | Speedup | Measured order |
|---|---|---|---|
| Euler–Maruyama (SDE) | | 1.0× | |
| Euler (ODE) | | | |
| Heun | | | |
| Exponential integrator | | | |
| Adaptive | | | |

**Stack.** PyTorch + `diffusers`, a small pretrained model (MNIST/CIFAR-class DDPM, or a compact
Stable Diffusion if you have the GPU), NumPy/SciPy.

---

## Part B — the honest solver Pareto (optional)

If you want the classical PDE work in the portfolio too, do it as a **fair fight**, not a
demonstration.

Pick **one stiff problem** (a reaction–diffusion system, or Burgers' at low viscosity, or a
convection–diffusion problem with a boundary layer). Then put five methods on the same
accuracy-vs-compute axes:

- Adaptive Dormand–Prince (RK45) — explicit, will struggle on the stiff case
- Implicit BDF — the correct classical answer for stiffness
- A spectral method — if the solution is smooth, this will embarrass everything else
- A PINN
- A Fourier neural operator

**The mathematics you own:** stability regions and why the explicit method's step size collapses
on the stiff problem; local truncation error and adaptive step-size control; the difference
between solving *one* instance (classical wins) and solving a *family* of instances (the neural
surrogate amortizes training across a parameter sweep, and can win).

**The limitation you volunteer first.** PINNs frequently **lose** to a well-implemented classical
solver on exactly the problems they are demoed with. Say it out loud, then show the Pareto front:
where the neural surrogate genuinely wins (many-query parameter sweeps, inverse problems, sparse
noisy data with known physics) and where it loses (single high-accuracy forward solve).

**A PINN reproducing a textbook Burgers' solution with no baseline is the single most recognizable
toy project in the field.** Hiring managers see it monthly. The baseline is what makes it real.

---

## Interview claim

> Your image model's sampler is an ODE solver. I cut the step count by N× at equal quality by
> replacing it with a higher-order method, and I can show the convergence-order plot that proves
> the implementation is correct.

## Suggested repo layout

```
diffusion-as-ode/
  README.md              <- log-log convergence plot + NFE table at the top
  src/
    samplers/
      euler_maruyama.py
      euler_ode.py
      heun.py
      exponential.py
      adaptive.py
    schedule.py          noise schedule / time parameterization
    reference.py         high-NFE reference solution
  analysis/
    convergence_order.py <- the credibility anchor
    nfe_vs_quality.py
  pde/                   <- Part B, optional
    classical/           rk45, bdf, spectral
    neural/              pinn, fno
    pareto.py
  results/
    convergence.png
    nfe_frontier.png
```

## References

- Song, Y. et al. [*Score-Based Generative Modeling through Stochastic Differential
  Equations*](https://arxiv.org/abs/2011.13456) (arXiv:2011.13456) — the paper that makes the
  ODE/SDE identification explicit. **Read this first.**
- Lu, C. et al. [*DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in
  Around 10 Steps*](https://arxiv.org/abs/2206.00927) (arXiv:2206.00927), and
  [DPM-Solver++](https://arxiv.org/abs/2211.01095) (arXiv:2211.01095) — the exponential integrator.
- Karras, T. et al. [*Elucidating the Design Space of Diffusion-Based Generative
  Models*](https://arxiv.org/abs/2206.00364) (arXiv:2206.00364) — Heun sampler, and a very clean
  treatment of the design choices as numerical ones. Code: [NVlabs/edm](https://github.com/NVlabs/edm).
- Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I / II* (Springer) — book, no
  free copy. Your actual authority on order, stability and stiffness. Cite it; almost nobody in ML
  does.
- [Physics-informed neural networks for differential equations: a comprehensive review](https://www.sciencedirect.com/science/article/pii/S0925231226007149)
- [Benchmarking neural surrogates on realistic spatiotemporal multiphysics flows](https://arxiv.org/pdf/2512.18595)
