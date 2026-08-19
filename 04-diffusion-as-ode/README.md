# 04 · Diffusion Samplers as Numerical Integrators

> Sampling from a diffusion model is solving a differential equation, so I measured the
> samplers the way a numerical analyst measures integrators: order of convergence,
> against an exact solution, on a log–log fit. Euler comes out at 0.99, Heun at 2.02,
> DPM-Solver-2 at 2.01 — and at matched *trajectory* accuracy the second-order and
> adaptive methods get there in **8–10× fewer network evaluations** (2× on a
> distributional metric, where a finite sample sets the floor; both tables are below).

**Status:** everything below runs and is tested with numpy alone; 54 tests, ~7 s;
`make results` regenerates every CSV in ~35 s. No pretrained model, no images and no
FID — see [STATUS.md](STATUS.md), which says exactly what is missing and why.

---

## The headline result

**Measured empirical order of convergence.** Step size halved eight times **on a grid
uniform in log-SNR `λ = log(α/σ)`** — the grid DPM-Solver and EDM actually use —
trajectory error taken against the *exact* solution (not a fine-grid reference — see
[the mathematics](#2-ground-truth-without-a-reference-solver)), slope fitted by least
squares over `N = 32 … 1024`. Regenerate with `make results`; raw numbers in
[`results/convergence.csv`](results/convergence.csv).

| Sampler | Type | NFE/step | Design order | **Fitted slope** | Fit residual (decades) |
|---|---|---|---|---|---|
| Euler | prob.-flow ODE | 1 | 1 | **0.993** | 0.002 |
| Heun | prob.-flow ODE | 2 | 2 | **2.017** | 0.004 |
| DPM-Solver-1 (exponential) | prob.-flow ODE | 1 | 1 | **0.996** | 0.001 |
| DPM-Solver-2 (exponential) | prob.-flow ODE | 2 | 2 | **2.015** | 0.004 |
| Euler–Maruyama, strong | reverse SDE | 1 | 1 (not ½ — see below) | **1.001** | 0.005 |
| Euler–Maruyama, weak on `E[X²]` | reverse SDE | 1 | 1 | **1.127** | 0.028 |

The CSV also carries the same fits on a grid **uniform in `t`**, where the numbers are
1.009 / 1.992 / 0.995 / **1.931**. Everything is second order there too except
DPM-Solver-2, which reads 1.93 — because a uniform-`t` grid makes its λ-steps span a
500× range (`dλ/dt = −β/2σ²` varies that much across `[10⁻³, 1]`), so it is not yet
asymptotic in its own natural variable at `N = 1024`. Both grids are in
`results/convergence.csv`; the table above is the log-SNR one, and this paragraph is
here so that a reader diffing the README against the CSV does not have to guess which.

The per-interval orders `log₂(e_h / e_{h/2})` settle onto the integers rather than
averaging to them, which is the part that is hard to fake:

| N | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| Euler | 0.84 | 0.95 | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 |
| Heun | 1.96 | 2.08 | 2.04 | 2.02 | 2.01 | 2.01 | 2.00 |
| DPM-Solver-2 | 2.05 | 2.07 | 2.04 | 2.02 | 2.01 | 2.00 | 2.00 |

**And the cost table the brief asks for.** NFE needed to reach a fixed accuracy, and the
speedup over first-order Euler on the probability-flow ODE
([`results/nfe_targets.csv`](results/nfe_targets.csv)):

| Sampler | NFE for W₁ ≤ 0.02 | speedup | NFE for trajectory RMSE ≤ 3·10⁻³ | speedup |
|---|---|---|---|---|
| Euler–Maruyama (SDE) | 64 | 0.75× | — (no pathwise counterpart) | — |
| Euler (ODE) | 48 | 1.0× | 384 | 1.0× |
| **Heun** | **24** | **2.0×** | **48** | **8.0×** |
| DPM-Solver-1 | 128 | 0.38× | not reached by 512 | — |
| DPM-Solver-2 | 48 | 1.0× | 128 | 3.0× |
| **Adaptive (PI-controlled Heun)** | **26** | **1.85×** | **38** | **10.1×** |

The brief's column heading is *NFE for FID ≤ target*. FID needs an Inception network and
a real image model; neither is available here, so the column is the metric I can
actually compute exactly — 1-Wasserstein against the known target law — and the
substitution is named rather than hidden.

**Two results in that table are not the ones I expected, and they are the interesting
ones.** DPM-Solver-1 is *worse* than plain Euler on this problem, and DPM-Solver-2 is
worse than Heun. That is not a bug: the exponential integrator's advantage is a function
of how close the data distribution is to a point mass, and I measured the crossover —
see [§5](#5-why-the-exponential-integrator-wins-and-when-it-does-not).

---

## The problem

The reverse-time process of a score-based generative model is an SDE

```
dx = [ f(x,t) − g(t)² ∇ₓ log p_t(x) ] dt + g(t) dW̄
```

and its deterministic counterpart, the probability-flow ODE, has the same marginals:

```
dx/dt = f(x,t) − ½ g(t)² ∇ₓ log p_t(x)
```

The network *is* the vector field. "Number of sampling steps" is *step count*.
"Sampler" is *integration scheme*. So the questions a numerical analyst asks — what is
the order, what is the error constant, where is the stability boundary, when does an
exponential integrator beat an explicit one — are the questions that decide how many
GPU-seconds an image costs.

**The design decision that makes this verifiable.** There is no neural network anywhere
in this repo. The prior is a Gaussian mixture, and under a linear forward SDE the
perturbed marginal `p_t` is *again* a Gaussian mixture with known parameters, so
`∇ₓ log p_t` is available in closed form. **The network is replaced by its exact analytic
counterpart precisely so that the numerical claims are about the integrator and nothing
else.** No training run, no seed variance, no approximation error of unknown size —
every error reported here is a discretization error, and the exact answer it is compared
against is exact.

---

## The mathematics

### 1. The forward SDE and the closed-form score

The variance-preserving SDE, `dx = −½β(t)x dt + √β(t) dw` with
`β(t) = β_min + t(β_max−β_min)`, has a Gaussian transition kernel. Writing
`B(t) = ∫₀ᵗβ`,

```
α(t) = exp(−B(t)/2),      σ(t)² = 1 − exp(−B(t)),      α² + σ² = 1
```

so a prior `p₀ = Σ w_k N(μ_k, v_k I)` perturbs to

```
p_t = Σ_k w_k N( α(t) μ_k , V_k(t) I ),    V_k(t) = α(t)² v_k + σ(t)²
```

and differentiating `log p_t` gives the score in one line:

```
∇ log p_t(x) = − Σ_k r_k(x,t) (x − α μ_k) / V_k(t),
r_k = softmax_k( log w_k + log N(x; α μ_k, V_k) )
```

The softmax is evaluated by subtracting the row maximum — with well-separated components
the raw densities underflow long before the responsibilities do.
→ [`src/sde.py`](src/sde.py), checked against finite differences of `log p_t` to 1e-6 in
[`tests/test_sde.py`](tests/test_sde.py).

### 2. Ground truth without a reference solver

A convergence study is only as good as what it measures error against, and a fine-grid
numerical "reference" biases exactly the finest levels that decide the fitted slope.
Both exact solutions used here are derived in [`src/reference.py`](src/reference.py).

**(a) A single Gaussian, any dimension.** Guess `x(t) = m_t + √(V_t/V_s)(x(s) − m_s)`
and substitute; matching the terms in `(x − m)` reduces the claim to
`V′ = 2 a(t) V + g(t)²`, which for VP is `V′ = β(1−V)` — and
`V = 1 − e^{−B}(1−v)` gives `V′ = β e^{−B}(1−v) = β(1−V)`. So the probability-flow map
for a Gaussian prior is *affine*, and known.

**(b) Any 1-D prior: the flow map is the quantile map.** The probability-flow ODE
transports `p_s` to `p_t` by construction; its field is Lipschitz in `x`, so solutions
are unique and trajectories cannot cross; in one dimension a non-crossing flow map is
strictly increasing; and a strictly increasing map pushing `p_s` to `p_t` is unique — it
is the quantile map. Hence

```
Φ_{s→t}(x) = F_t⁻¹( F_s(x) )       exactly, for a mixture with any number of components.
```

That is the ground truth every order fit in this repo uses. Two consequences worth
stating:

- **Initial conditions are specified by probability levels, not by `x` values.** Then
  `x(s) = F_s⁻¹(p_i)` and the exact answer is `x(t) = F_t⁻¹(p_i)` — no forward CDF is
  ever evaluated. `F_s(x)` saturates to 0 or 1 in double precision a few standard
  deviations out, and inverting a saturated value would manufacture error that has
  nothing to do with the integrator.
- **The two ground truths check each other.** For a Gaussian prior, (a) and (b) are
  independent derivations of the same map and agree to 1e-10. And the *high-NFE*
  reference (DPM-Solver-2, 4096 steps) converges to (b) at ratio exactly **4.00 per
  doubling** — a second-order sampler cannot converge at order 2 towards a wrong map.
  → [`tests/test_reference.py`](tests/test_reference.py)

### 3. Local vs. global truncation error

Euler's local error is `½h²x″(τ)`; there are `N = O(1/h)` steps, and errors already
committed are transported by the flow, so the standard Gronwall argument gives

```
‖e_N‖ ≤ (hC / 2L) ( e^{L(T−t_eps)} − 1 )  =  O(h)
```

— order `p` locally is order `p` globally, one power of `h` spent on the step count.
The constant is the part that matters in practice, and it is not free: **`C` is
proportional to the conditioning of the flow map.** Differentiating
`F₀(Φ(x)) = F_T(x)`,

```
Φ′(x) = f_T(x) / f₀(Φ(x))
```

the ratio of source and target densities at corresponding points. Between two
well-separated modes the target density is astronomically small and `Φ′` is enormous:
the trajectory threading that gap is exponentially ill-conditioned. That is a property
of the *problem*, not of any integrator — and it is why the repo carries two priors.

| Prior | max \|Φ′\| | fitted Euler slope | fitted Heun slope | fit residual |
|---|---|---|---|---|
| canonical (modes ~4σ apart) | 4.6 | **0.993** | **2.017** | 0.002 / 0.004 |
| sharp (modes ~8σ apart) | 1.7·10⁴ | 1.341 | 1.956 | **0.172** / 0.004 |

Same code, same range of `h`, and the first-order slope reads 1.34 on the sharp prior —
because at `h` that large one is not in the asymptotic regime at all, and individual
trajectories cross between basins. The residual column is what gives it away: 0.172
decades means the points are not on a line, so the slope is not a slope. **Reporting the
1.34 as a measurement would be wrong; hiding it would be worse.** All the order fits in
the headline are on the canonical prior, over a stated window, with the residual quoted.
→ [`src/problem.py`](src/problem.py), [`analysis/convergence_order.py`](analysis/convergence_order.py)

### 4. Euler–Maruyama's strong order is 1, not ½

The textbook figure for Euler–Maruyama is strong order ½. That ½ comes from the Milstein
correction `½ g g_x (dW² − dt)`, which exists only when the diffusion coefficient depends
on the state. **In every diffusion model's reverse SDE it does not:** `g = g(t)` alone,
the noise is additive, the Milstein term vanishes identically, and Euler–Maruyama *is*
Milstein. Its strong order is 1. (Standard for additive-noise SDEs — Kloeden & Platen,
*Numerical Solution of Stochastic Differential Equations*.)

Measured slope **1.001**, residual 0.005 decades, with common random numbers: one
Brownian path is drawn on the finest grid and *summed* into every coarser grid, so all
resolutions are driven pathwise by the same path. That refinement is itself asserted to
machine precision in [`tests/test_samplers.py`](tests/test_samplers.py) before any slope
is fitted — a bug there produces a plausible-looking wrong number.

Weak order is reported on `E[X²]` (slope 1.13), not on `E[X]`. The mean's weak-error
constant on this problem is ~30× smaller and sits *below* the Monte-Carlo floor at 8192
paths; its "slope" of 1.55 with residual 0.21 decades is noise, and the CSV carries the
floor next to it so the reader can see that. The mean's weak order is instead pinned
exactly: for a Gaussian prior the reverse SDE is linear with additive noise, so
`E[X_{n+1}] = E[X_n](1+hA_n) + hb_n` is deterministic — running the sampler with zero
Brownian increments *is* the mean — and that recursion fits slope 1.000 with no
Monte-Carlo error at all.

### 5. Why the exponential integrator wins — and when it does not

The probability-flow ODE is **semi-linear**. In the noise parameterization,

```
dx/dt =  a(t) x  +  ( g(t)² / 2σ(t) ) ε(x,t)
         \_____/    \____________________/
         linear,    nonlinear, and smooth: ε stays O(1) as t→0 while the
         stiff,     score it is built from blows up like 1/σ
         exact
```

Variation of constants with the integrating factor `α(t)`, then a change of variable to
the log-SNR `λ = log(α/σ)` (for which `dλ/du = −g²/2σ²`, so `ασ dλ = −(g²/2) du`), gives

```
x(t) = (α_t/α_s) x(s) − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂(λ) dλ
```

Everything except `ε̂` is now exact, and the `e^{−λ}` weight — which is where the
stiffness lives — is integrated in closed form. Freezing `ε̂` at the left endpoint is
DPM-Solver-1; evaluating it at the λ-midpoint is DPM-Solver-2 (Lu et al., 2206.00927).
Integrating the linear part exactly and approximating only the remainder is a classical
numerical-analysis idea, not a deep-learning one.

**When is one step already exact?** Exactly when `ε̂` is constant along the trajectory.
For a **point-mass prior** it is: `p_t = N(αμ, σ²)`, the flow is
`x(t) = α_t μ + (σ_t/σ_s)(x(s) − α_s μ)`, so `ε = (x − αμ)/σ` never moves. One step —
the whole interval, in a single evaluation — reproduces the analytic solution to
**6·10⁻¹³**, which is machine precision for a computation whose intermediate quantities
reach 456; at two or more steps the cancellation disappears and the residual is 2·10⁻¹⁵.
Euler and Heun are off by more than 10⁻² on the same step.

For a Gaussian prior of variance `v > 0` it is *not* exact, and the deficit is closed
form. Both schemes multiply `x(s) − α_s μ` by a scalar:

```
R_exact = √(V_t/V_s)        R_dpm1 = (α_t α_s v + σ_t σ_s) / V_s
R_exact² − R_dpm1²  =  v (α_t σ_s − σ_t α_s)² / V_s²
```

so the error is proportional to the prior variance times the squared log-SNR gap, and
vanishes iff `v = 0` or the step is empty. That identity is asserted against the measured
one-step error to 11 decimal places in
[`tests/test_exponential.py`](tests/test_exponential.py) — a sharper statement about the
implementation than "one step is exact", because it pins the entire `v`-dependence.

**And it explains the two surprising rows in the headline table.** Sweeping the prior
variance at 16 fixed steps against the exact solution
([`results/stability.csv`](results/stability.csv)):

| prior variance `v` | 1.0 | 0.1 | **0.01** | 10⁻³ | 10⁻⁴ | 0 |
|---|---|---|---|---|---|---|
| Euler | 3.1e-2 | 2.8e-2 | 1.4e-2 | 7.1e-3 | 5.7e-3 | 5.8e-3 |
| DPM-Solver-1 | 1.4e-1 | 4.4e-2 | 1.4e-2 | 4.2e-3 | 1.0e-3 | **1.1e-16** |
| Euler / DPM-1 | 0.22 | 0.64 | **1.00** | 1.69 | 5.77 | ∞ |

The crossover is at `v ≈ 0.01`. Above it, freezing the ODE field beats freezing `ε`;
below it, the reverse. Real image data sits far down that axis — the data manifold is
thin, so `p₀` is close to a sum of point masses — which is why DPM-Solver wins in
production and why it does not win on a broad Gaussian mixture. **This is the result I
would lead with in an interview**, because it is the mechanism, not the leaderboard.

### 6. Stiffness — and the step-size restriction that turns out not to bite

The reverse ODE's Jacobian `J = ∂F/∂x` diverges as `t → 0`. For a single Gaussian,
`J = β(t)(1−V)/(2V) > 0`, which grows like `1/(2t)` (because `V ≈ β_min t` there),
capped at `β_min/(2v)` once the data's own variance dominates. So the process is stiff,
as the folklore says. I went looking for the step-size restriction that is supposed to
follow, and **it is not the binding constraint.**

**The sign is the whole argument, and it is easy to lose in an absolute value.**
Sampling integrates backwards, so `h < 0`.

- Where `J > 0` the step is *contracting*, `hJ < 0`, and explicit Euler's amplification
  factor `1 + hJ` leaves `[−1, 1]` as soon as `|h| J > 2`. That is the classical
  restriction.
- Where `J < 0` — which happens *between* the modes of a mixture, since
  `J = −β/2 − (β/2)∂²log p/∂x²` and `log p` is convex in the trough — the step is
  *expanding*, `1 + hJ > 1`, and there is no bound to violate: the exact linearized
  factor `e^{hJ}` is larger still.

So [`results/stability.csv`](results/stability.csv) records both signed extremes, not
just `max|hJ|`. Measured on both priors, both grid families, `N = 4 … 128`:

| steps | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| canonical, `max\|hJ\|` | 0.38 | 0.55 | 0.35 | 0.20 | 0.11 | 0.06 |
| canonical, `min(1+hJ)` | +0.62 | +0.69 | +0.77 | +0.87 | +0.93 | +0.96 |
| sharp, `max\|hJ\|` | 1.27 | **4.66** | 2.33 | 1.74 | 0.87 | 0.44 |
| sharp, `min(1+hJ)` | +0.32 | **+0.18** | +0.27 | +0.52 | +0.76 | +0.88 |

`min(1+hJ)` never falls below **+0.18**. **Explicit Euler is never outside its stability
region on this schedule, at any step count I measured.** The `1/(2t)` growth of `J` is
almost exactly cancelled by the fact that a monotone grid ending at `t_eps` cannot take
a step larger than `t` itself.

What is there instead, on the sharply multimodal prior at 8 steps, is the opposite
failure. `max hJ = +4.66` — an *expanding* step — so a single step must reproduce a
linearized amplification of `e^{4.66} = 106`, and explicit Euler reproduces `1 + 4.66 =
5.66`. **A 19× under-estimate in one step.** That is an accuracy catastrophe, not an
instability, and it is precisely why the order fits on the sharp prior in §3 are
meaningless at those step counts. Getting this the wrong way round — quoting 4.66 as a
violated stability bound — would have been an easy and wrong story to tell.

The exponential integrator sidesteps the question entirely. Its factor
`R_dpm1 = (α_tα_s v + σ_tσ_s)/V_s` is positive term by term and, by the identity in §5,
never exceeds `R_exact ≤ 1`. So it lies in `(0, R_exact]` for **every** step size,
including one step across the whole interval: unconditionally stable, never oscillating,
never overshooting. Verified down to a single step in `results/stability.csv`, where
Euler's amplification factor is meanwhile wrong by a factor of 29.

### 7. Adaptive step size

Heun's two stages already contain an order-1 result, so the embedded error estimate
`x_heun − x_euler = (h/2)(k₂−k₁)` is free. We advance with the higher-order member
(local extrapolation), scale componentwise by `atol + rtol·max(|x|,|x̂|)`, and drive the
step size with Gustafsson's PI controller as given in Hairer & Wanner,

```
h_new = h · safety · err_n^{−0.7/k} · err_{n−1}^{+0.4/k},   k = 2
```

Rejected steps cost their two evaluations and are counted; a solver that hid them would
not be comparable with a fixed-step one. One step size is shared across the batch,
because the score is evaluated on the batch in one forward pass — the same choice
production samplers make. → [`src/samplers/adaptive.py`](src/samplers/adaptive.py)

**The honest baseline is a grid uniform in log-SNR**, not uniform in `t`. On the VP
schedule `dλ/dt` varies by a factor of ~500 across `[10⁻³, 1]`, so beating a uniform-`t`
grid is easy and means nothing; uniform-λ is what DPM-Solver and EDM actually use.
Against *that*, adaptive stepping reaches trajectory RMSE ≤ 3·10⁻³ in **38 NFE where
fixed-step Heun needs 48**. The advantage narrows as the tolerance tightens — matching
each adaptive run against the cheapest fixed-step Heun that reaches the same error, on
the budgets in [`results/nfe_quality.csv`](results/nfe_quality.csv), gives 26 vs 48 at
the loose end and 334 vs 384 at the tight end — because a uniform-λ grid is already
close to optimal for this problem. The controller's requested tolerance is honoured at
every setting (final error below `rtol` in all eight runs), though a local error control
is not a global error bound and I would not claim it is.

---

## Running it

No installation, no downloads, no network. numpy is the only requirement.

```bash
make test        # 54 tests, ~7 s
make results     # regenerates every CSV in results/, ~35 s
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

The suite contains five assertions marked in the source with
`=== THE TEST THAT MATTERS ===`, each of which fails if the mathematics is wrong rather
than if the code crashed:

- **The fitted log–log slopes are the design orders**, ±0.1, with the straightness of
  the fit asserted separately (residual < 0.02 decades). A wrong sign in the score, a
  missing ½ on `g²`, a mis-derived `dλ/dt` or a midpoint at the wrong time all still run
  and still produce plausible samples — and all move the slope off its integer.
  → [`tests/test_convergence.py`](tests/test_convergence.py)
- **The probability-flow ODE reproduces the analytic marginal.** Integrating `T → t_eps`
  from the exact quantiles of `p_T` must return the analytic mean and variance of
  `p_{t_eps}`; tolerances are set from the measured quadrature floor of the initial
  ensemble (8·10⁻⁶ in the mean, 4.5·10⁻⁴ in the variance), not tuned.
  → [`tests/test_samplers.py`](tests/test_samplers.py)
- **The exponential integrator is exact in one step on a point mass**, and its deficit
  for `v > 0` matches the closed form to 11 places.
  → [`tests/test_exponential.py`](tests/test_exponential.py)
- **The high-NFE reference converges to the analytic quantile map at ratio 4.00 per
  doubling** — which it could not do if either the transport argument or the sampler
  were wrong. → [`tests/test_reference.py`](tests/test_reference.py)
- **Euler–Maruyama's strong order is 1, not ½**, asserted as 1 with the additive-noise
  argument of §4 written out beside it — and the Brownian coarsening it relies on is
  checked to machine precision first, because a bug there yields a plausible wrong
  slope. → [`tests/test_convergence.py`](tests/test_convergence.py)

Plus: Euler–Maruyama's weak order on the mean is 1, pinned without Monte-Carlo noise;
adaptive stepping honours its tolerance and beats uniform-λ Heun at matched accuracy;
NFE counts are exactly the advertised rate per step; the metrics reproduce known answers
(`W₁` between `P` and `P + c` is `|c|`, the closed-form energy-distance terms match
brute-force pairwise sums); and — since matplotlib cannot run here — every CSV column
`analysis/plot_results.py` indexes by name is asserted to exist, so a typo in the
unrunnable file still fails the suite.

---

## Results

### Accuracy per NFE ([`results/nfe_quality.csv`](results/nfe_quality.csv))

Trajectory RMSE against the exact flow map — pure discretization error, no statistical
floor:

| NFE | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| Euler (ODE) | 6.7e-2 | 3.4e-2 | 1.7e-2 | 8.8e-3 | 4.4e-3 | 2.2e-3 |
| Heun | 2.7e-2 | 7.0e-3 | 1.7e-3 | 4.0e-4 | 1.0e-4 | 2.5e-5 |
| DPM-Solver-2 | 1.2e-1 | 2.9e-2 | 6.9e-3 | 1.7e-3 | 4.2e-4 | 1.0e-4 |

Every sampler is given the same 8192 starting points — the midpoint quantiles of
`N(0,1)`, so the input ensemble contributes no Monte-Carlo error of its own — and the
same grid family. The floor is measured, not assumed: pushing those inputs through the
exact map gives `W₁ = 4.4·10⁻⁵`, which is the prior-mismatch error of starting from
`N(0,I)` rather than the true `p_T` (`α_T = 6.6·10⁻³`, so they are close but not equal).

### The stochastic/deterministic tradeoff ([`results/sde_vs_ode.csv`](results/sde_vs_ode.csv))

Both processes have **the same marginals** — that is the theorem — so "the SDE is more
diverse" cannot be a statement about the marginal. What differs:

**Accuracy per NFE.** `W₁` against the exact `p_{t_eps}`:

| NFE | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| Euler–Maruyama (SDE) | 0.045 | 0.024 | 0.018 | 0.016 | 0.012 | 0.012 |
| Euler (ODE) | 0.047 | 0.023 | 0.012 | 0.0059 | 0.0029 | 0.0015 |
| Heun (ODE) | 0.023 | 0.0066 | 0.0016 | 3.9e-4 | 1.1e-4 | 5.2e-5 |

The ODE keeps converging; the SDE stalls around 0.012, because it injects fresh
randomness at every step and a finite sample of a random map carries `O(n^{-1/2})` error
however well the SDE is integrated. The Monte-Carlo floor of 8192 exact i.i.d. draws is
`W₁ = 0.0203`, measured.

**Conditional diversity.** Fix a state `x_{t_c}` and sample to `t_eps` many times. The
ODE is a deterministic map, so the spread is *exactly* zero at every `t_c`. The SDE's
conditional mode entropy, against the prior's own 1.559 bits:

| conditioning time `t_c` | 1.0 | 0.6 | 0.4 | 0.25 | 0.15 | 0.08 | 0.04 |
|---|---|---|---|---|---|---|---|
| SDE, bits | 1.56 | 1.55 | 1.49 | 1.20 | 0.40 | 0.02 | 0.00 |
| ODE, bits | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Started at `t_c = T`, the reverse SDE **forgets its initialization completely** — the
conditional entropy equals the marginal entropy. That is the quantity people mean by
diversity, and it is conditional, never marginal.

**And what the ODE buys instead: invertibility.** `T → t_eps → T` with Heun returns to
the start with RMSE 4.3e-4 at 256 NFE and 6.8e-6 at 512, converging at order 2. The SDE
map has no inverse, so that column is blank rather than filled with something
meaningless.

---

## The limitation I volunteer first

**There is no neural network in this repository, and no images.** The score is analytic.
That was a deliberate trade — it is what lets me claim a *measured* order of convergence
against an *exact* solution rather than against a fine-grid guess — but it costs
something real, and the cost is this: I have shown that my integrators have the orders
they should and that their relative cost behaves as the theory predicts, on a problem
where the vector field is smooth and known. A trained score network is neither. It has
approximation error that does not shrink with `h`, it is only piecewise smooth in `t`,
and near `t → 0` it is fit on data the model has effectively never seen. Every one of
those degrades a high-order method more than a low-order one, because high order buys you
nothing once the truncation error is below the network's own error. **The honest claim is
that the integrators are correct and the mechanism is understood — not that an 8× NFE
reduction transfers to Stable Diffusion.** I would expect a real speedup between 2× and
4×, which is what the DPM-Solver and EDM papers report, and I would want to measure it
before saying so.

**Second: the exponential integrator loses on my test problem.** DPM-Solver-1 needs 128
NFE where Euler needs 48. I could have picked a sharper prior and shown it winning; I
measured the crossover instead (§5) and reported that it sits at prior variance ≈ 0.01,
below my canonical problem. The mechanism generalizes; the leaderboard position does not.

**Third: the SDE numbers are noisier than the ODE numbers, and I have said where.**
Euler–Maruyama's weak order on the mean is at the Monte-Carlo floor at 8192 paths and I
report the floor beside it rather than a clean-looking slope. The strong-order fit is
solid (residual 0.005 decades) because common random numbers remove the noise from a
pathwise comparison; nothing similar exists for a weak error.

**Fourth: this is one dimension.** The exact quantile transport that makes the ground
truth exact is a one-dimensional argument — monotone transport is unique on the line and
nowhere else. The samplers, the schedule and the score are dimension-agnostic and the
affine Gaussian solution holds in any dimension, but the sharpest ground truth does not,
and a real image model is 10⁵-dimensional. Part B of the brief (the classical-vs-neural
PDE Pareto front) needs torch and is not built — see [STATUS.md](STATUS.md).

---

## References

- Song, Y. et al. [*Score-Based Generative Modeling through Stochastic Differential
  Equations*](https://arxiv.org/abs/2011.13456) (arXiv:2011.13456) — the SDE/ODE
  identification, the VP and VE families, and Anderson's reverse-time SDE as used here.
- Lu, C. et al. [*DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
  Sampling in Around 10 Steps*](https://arxiv.org/abs/2206.00927) (arXiv:2206.00927) and
  [DPM-Solver++](https://arxiv.org/abs/2211.01095) (arXiv:2211.01095) — the exponential
  integrator, rederived from variation of constants in
  [`src/samplers/exponential.py`](src/samplers/exponential.py).
- Karras, T. et al. [*Elucidating the Design Space of Diffusion-Based Generative
  Models*](https://arxiv.org/abs/2206.00364) (arXiv:2206.00364) — the Heun sampler, and
  the cleanest treatment of these choices as numerical ones. Code:
  [NVlabs/edm](https://github.com/NVlabs/edm).
- Hairer, E., Nørsett, S. P. & Wanner, G. *Solving Ordinary Differential Equations I / II*
  (Springer) — book, no free copy. The authority for order, the PI step-size controller,
  stability regions and stiffness. Almost nobody in ML cites it.
- Kloeden, P. E. & Platen, E. *Numerical Solution of Stochastic Differential Equations*
  (Springer) — book. The additive-noise strong-order-1 result for Euler–Maruyama in §4.

Full brief: [BRIEF.md](BRIEF.md).
