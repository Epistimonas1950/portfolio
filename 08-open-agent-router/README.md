# 08 · A Cost-Optimal Agent Over a Fleet of Free Open Models

> An agent that routes every call to the cheapest model that can handle it — with the
> routing policy's regret exponent measured against the bound, and the escalation rule
> replaced by a conformal test that carries a finite-sample coverage guarantee.

**Status:** the mathematics runs, is tested, and produces every number below on a
**simulated fleet**. No language model of any kind runs on this machine — that is a
deliberate choice of instrument, argued in [the next paragraph](#what-cost-means-when-every-model-is-free)
and honestly bounded in [STATUS.md](STATUS.md).

---

## What "cost" means when every model is free

Every model in this fleet is open-weight and runs on hardware you already own, so there
are no dollars, and "cost-optimal" is meaningless until you say what you are minimising.
The quantity a company actually pays for is **occupancy of the accelerator**: wall-clock
seconds per call, optionally weighted by how much of the device the model's resident
weights hold. That is the unit everywhere in this repo, and it lives behind one interface
in [`src/cost.py`](src/cost.py) — `CostModel.price(CallCost) -> float`. Swapping to a
hosted fleet billed per token means constructing a `TokenPriceCostModel` and changing
nothing in `src/routers/`. The bandit never learns what a second is; it learns a scalar
called cost.

**And why the fleet is simulated.** There is no Ollama here, no llama.cpp, no vLLM, no
weights, no GPU and no network. Rather than mock a fleet, I built one whose ground truth
is known ([`src/fleet/simulator.py`](src/fleet/simulator.py)) — a latent per-query
difficulty drives each arm's success probability and its cost along a genuine capability
ladder. That is the right instrument, not a workaround, for three reasons:

1. **The oracle policy is computable, so regret is measured rather than estimated.**
   Regret is defined against the best achievable policy. On a real fleet you can only
   estimate that, so "regret" becomes a number you fit. Here every arm's success
   probability is known in closed form and the oracle is exact.
2. **Coverage can be checked against a known conditional distribution** — including by
   breaking exchangeability on purpose and watching the guarantee fail.
3. **The capability ladder is a controlled variable.** With all arms equally good there
   is nothing to route; with one arm dominating there is nothing to learn.

[`src/fleet/client.py`](src/fleet/client.py) is the same interface over a real server and
raises an error naming exactly what it needs. The routers cannot tell the difference.
**Every number below is labelled with the instrument that produced it.**

---

## The headline result

### 1. The regret exponent, on a synthetic linear-reward bandit

`Õ(d√T)` is a **minimax** bound — a statement about the worst instance at each horizon,
not about any one instance. Measuring it therefore means measuring a supremum over an
instance family indexed by a gap scale `Δ`:

```
R*(T)  =  max over Δ  of  E[ cumulative regret at horizon T on instance Δ ]
```

On the **synthetic linear-reward bandit** (`d = 20`, 5 arms, σ = 1, 12 gap scales,
5 seeds, T = 64,000; [`results/regret.csv`](results/regret.csv)):

| Policy | fitted exponent | R² of the log–log fit | prefactor |
|---|---|---|---|
| **LinUCB** | **0.468** | 0.9928 | 7.89 |
| Thompson sampling | 0.460 | 0.9988 | 9.95 |
| LinGreedy (no exploration bonus) | 0.470 | 0.9994 | 7.47 |
| **Random (control)** | **1.006** | 1.0000 | 2.00 |

Fitted over `t ∈ [2000, 64000]`, log-uniformly sampled. **LinUCB's exponent is 0.468
against the theory's ½, and the random policy's is 1.006 against 1** — the same
protocol, the same family, the same fitting code. The control is the evidence: if both
came out at ½ the measurement would be an artefact of the fit, and if both came out at 1
the bandit would not be learning.

The supremum is real rather than a grid edge: LinUCB's maximising gap is interior to the
family at every horizon and shrinks with `T` as the `Δ* ~ 1/√T` heuristic predicts
(Δ = 1.286 at `t = 2000`, Δ = 0.382 at `t = 64000`). The random policy's maximiser sits
on the upper edge — correctly, since its regret `c·Δ·T` is monotone in `Δ`, so moving the
boundary rescales the prefactor and leaves the exponent at 1.

LinUCB's R² is 0.9928 — the lowest of the four, against Thompson's 0.9988 and the random
policy's 1.0000 — for a structural reason worth naming rather than smoothing over. The
envelope is a maximum over a **discrete** grid of 12 gap scales, so it carries a small
kink wherever the maximiser steps from one grid point to the next. The random policy has
no such kinks because its maximiser never moves, which is exactly why its fit is perfect
and why a perfect fit here is not a sign of a better measurement. The kinks are an
artefact of the grid, not of the policy: the reduced configuration in the test suite uses
a different grid (8 scales, `d = 10`, `T = 16,000`) and lands on the same exponent.

**And the naive version of this experiment proves nothing, which is why it is also in the
CSV.** Fitting one *fixed* instance instead of the envelope, the local slope over five
sliding windows comes out

| fit window | local slope |
|---|---|
| `[100, 1000]` | 0.828 |
| `[316, 3160]` | 0.742 |
| `[1000, 10000]` | 0.623 |
| `[3160, 31600]` | 0.456 |
| `[6400, 64000]` | 0.377 |

A fixed instance has a smallest gap, and once the algorithm resolves it the regret turns
gap-dependent and logarithmic. Every fixed instance eventually leaves the √T regime, and a
curve bending from 1 towards 0 passes through ½ on the way — the window at
`[3160, 31600]` reads 0.456 and would let you announce that the theory is
confirmed, while the last window reads 0.377. Choosing `d`, `σ` and the fit window
lets you publish any exponent in (0,1), which is exactly why the headline measures a
supremum instead ([`results/regret.csv`](results/regret.csv), `experiment=fixed_instance`).

### 2. The cost–quality table, simulated fleet

20,000 queries × 5 seeds, three arms standing in for a 3B / 8B / 32B ladder with a
1 : 2.0 : 5.1 cost ratio ([`results/pareto.csv`](results/pareto.csv)). Reward is
`r = success − λ·cost` with `λ = 0.12`. All numbers **simulated fleet**.

| Policy | Task success | GPU-s / query | Peak mem (GB) | Escalation | Mean reward | Cost @ matched accuracy |
|---|---|---|---|---|---|---|
| Always smallest | 0.6974 | 0.547 | 2.2 | — | 0.6318 | 1.00× |
| Always largest | 0.9784 | 2.807 | 18.6 | — | 0.6416 | 1.00× |
| Random arm | 0.8479 | 1.483 | 18.6 | 0.667 | 0.6699 | 1.35× |
| **Difficulty-threshold classifier** | 0.9026 | 1.255 | 18.6 | 0.796 | **0.7521** | **0.45×** |
| LinUCB | 0.9076 | 1.327 | 18.6 | 0.851 | 0.7483 | 0.47× |
| LinGreedy (no exploration) | 0.7310 | 0.657 | 2.8 † | 0.200 | 0.6522 | 1.00× |
| Thompson sampling | 0.9008 | 1.287 | 18.6 | 0.870 | 0.7463 | 0.46× |
| Budgeted (single-price) | 0.8487 | 0.921 | 18.6 | 0.620 | 0.7383 | 0.84× |
| Conformal cascade (α = 0.2) | 0.9780 | 4.334 | 18.6 | 0.999 | 0.6499 | 1.54× |
| Oracle (cheapest sufficient) | 0.9845 | 0.868 | 18.6 | — | 0.8803 | — |
| Oracle (expected-reward) | 0.9177 | 1.262 | 18.6 | — | 0.7663 | — |

† LinGreedy is the only policy that never touches the 32B arm, so its provisioning
figure is 2.2 or 5.4 GB depending on the seed; 2.8 is the mean over five, and it is the
one cell in this column that is not a single number.

**LinUCB runs at 0.47× the per-query cost of the cheapest fixed arm that matches or beats
its success rate** — 2.1× less compute for at-least-equal quality, simulated fleet. (The
comparator is always-largest, which is *more* accurate at 0.978 against LinUCB's 0.908, so
the ratio is a conservative reading, not a flattering one.) The difficulty-threshold row
does slightly better still at 0.45×. And the row that matters more:

### 3. The simple baseline wins at 20,000 queries, and is never behind at 1.2 million

BRIEF.md predicts that a threshold on a cheap difficulty classifier often matches a
contextual bandit, and asks for that to be a first-class row rather than a footnote. On
this workload it does not merely match — **it beats LinUCB, and the margin survives its
error bar.** Paired over the five workload seeds (both policies see the identical
queries, so the pairing removes the workload variance):

```
mean reward difference, threshold − LinUCB  =  +0.00379  ±  0.00041   (9.2 standard errors)
```

That is a real ordering at `T = 20,000`, not noise, and it is published because it is
true. The obvious follow-up is whether it lasts, and that is answerable, because the two
policies accumulate regret at different rates
([`results/regret.csv`](results/regret.csv), fitted over `t ∈ [2000, 400,000]`):

| Policy on the simulated fleet | regret exponent | R² |
|---|---|---|
| LinUCB | 0.858 | 0.9968 |
| Thompson sampling | 0.861 | 0.9982 |
| Difficulty-threshold (frozen) | 0.999 | 1.0000 |
| Random | 0.999 | 1.0000 |

The threshold router's cut-points are frozen at fit time, so it pays a constant expected
regret per query for ever and its cumulative regret is **linear** — exponent 0.999.
LinUCB's is sublinear at 0.858. The obvious inference is that a sublinear curve must
eventually pass under a linear one, so the bandit wins at some horizon and the only
question is which. That inference is **wrong here**, and finding out why is worth more
than the crossing point would have been. Rather than extrapolate two fitted exponents I
extended the run to 1.2 million queries and looked.

> **It never overtakes.** Not at 400,000 queries and not at 1,200,000. The frozen
> classifier's cumulative regret is the lower of the two at every horizon measured, and
> the learner's curve never crosses it once.

The mechanism is the misspecification listed as this repo's third limitation, and it is
the most useful thing measured here. **LinUCB's exponent on the fleet is not a number, it
is a moving target.** Fitting the same curve over nested windows `[2000, H]`
([`results/regret.csv`](results/regret.csv), `experiment=fleet_long_horizon`):

| fit window | LinUCB exponent | frozen threshold exponent |
|---|---|---|
| `[2000, 40,000]` | **0.786** | 1.002 |
| `[2000, 120,000]` | **0.825** | 0.998 |
| `[2000, 400,000]` | **0.863** | 0.998 |
| `[2000, 1,200,000]` | **0.890** | 0.999 |

The frozen baseline sits at 0.999 and does not move — a policy that never learns accrues
a constant expected regret per query, so its cumulative regret is linear by construction
and its exponent is 1 at every horizon. LinUCB's climbs *toward* that same 1. That is
precisely what a misspecified model does: the part of the reward surface the features
cannot represent contributes a residual regret on every query, which no amount of data
removes, so the learner's regret is asymptotically linear too. It is not converging to a
√T advantage it can use to overhaul the baseline; it is converging to the same linear rate
with a slightly worse constant.

So the honest answer to "where does the bandit earn its complexity on this workload" is:
**it does not.** At best it ties, while carrying more machinery, more hyper-parameters
and more variance. Buy the classifier. BRIEF.md asks for exactly that to be published if
it turns out to be true, and here it holds over every horizon measured, from 2,000 to
1.2 million queries.

One caveat in the other direction, because the difference is smaller than it looks. The
paired final-regret gap at 1,200,000 is -395 ± 553 over
6 workload seeds — statistically indistinguishable. LinUCB's cumulative regret has
a heavy seed-to-seed spread (one seed in six lands about 20% above the others, which is
the bandit occasionally committing early to a bad estimate and paying for it all run),
and that spread is wide enough to swallow the mean difference. The defensible claim is
therefore the *sign and the exponent*, not the size: the frozen rule is never behind, and
the learner has no asymptotic mechanism to get ahead.

Two things the bandit still buys, which this experiment is deliberately constructed *not*
to reward, and which are the honest counterweight rather than a consolation:

- **It needs no labelled tuning set.** The threshold's cut-points were fitted on 6,000
  queries with *full information* — the reward every arm would have earned on every one
  of them. That is exactly the signal you do not have at serving time, and it is the
  advantage BRIEF.md warns about handing a baseline for free. The bandit learned from
  bandit feedback: one arm, one reward, per query.
- **It adapts, and the frozen rule cannot.** This workload is stationary by construction.
  Under drift the classifier's cut-points are wrong and stay wrong at any horizon, while
  the bandit's estimates move. Measuring that would need a non-stationary workload, which
  is not built here and is listed as absent in [STATUS.md](STATUS.md).

The comparison that is *not* close is against the arms themselves: both routers cut
regret by roughly 7× against a random arm and beat every fixed-arm policy by a wide
margin. The interesting fight is between the bandit and the cheap classifier, and the
cheap classifier is not losing it.

---

## The problem

An agent makes an LLM call at every step, and most systems call the largest model every
time. Choosing which model to call is a sequential decision under uncertainty with a cost
constraint — a contextual bandit, with theorems. At step `t` a query arrives with a
serving-time feature vector `x_t`, you choose an arm `a_t`, observe reward `r_t` and pay
cost `c_t`, with

```
r_t  =  quality_t  −  λ · cost_t
```

Two things then need to be true for the system to be defensible: the routing policy has
to converge at a rate you can state, and the escalation rule has to carry a guarantee
rather than a hand-tuned threshold. Those are the two claims above, and the mathematics
for each is below.

## The mathematics

### 1. Where LinUCB's confidence width comes from

Arm `a` has its own ridge model, `A_a = λI + X_aᵀX_a`, `θ̂_a = A_a⁻¹ bₐ`. The
self-normalised tail bound for vector-valued martingales gives, with probability `1 − δ`,
simultaneously for all `t`,

```
‖θ̂_a − θ_a‖_{A_a}  ≤  σ √( 2 log(1/δ) + log( det A_a / det λI ) )  +  √λ ‖θ_a‖₂  =:  β_t
```

and the `A_a`-weighted Cauchy–Schwarz inequality transfers that from the parameter to the
direction we actually care about:

```
| xᵀθ̂_a − xᵀθ_a |  ≤  ‖x‖_{A_a⁻¹} · ‖θ̂_a − θ_a‖_{A_a}  ≤  β_t √( xᵀ A_a⁻¹ x )
```

So the *shape* of the interval is forced — it is the Mahalanobis length of the query
direction under the accumulated design. A direction an arm has been probed in often is
narrow; an unexplored one is wide. That is the entire content of "optimism". With
`‖x‖ ≤ 1` the determinant grows at most like `(λ + t/d)^d`, so
`β_t = O(σ√(d log t) + √λ S)`, and the elliptical-potential argument gives `Õ(d√T)`.
In practice `β_t` is replaced by a constant `α`; that deviation is stated in the code
rather than hidden, and the exponent is what this repo verifies, not the constant.
→ [`src/routers/linucb.py`](src/routers/linucb.py)

**Sherman–Morrison.** Each round updates `A_a` by a rank-one term. Re-inverting is
`O(d³)` per round; the identity

```
(A + xxᵀ)⁻¹  =  A⁻¹  −  (A⁻¹x)(A⁻¹x)ᵀ / (1 + xᵀA⁻¹x)
```

updates the inverse in `O(d²)` — a factor `d`, which at `T = 64,000` is the difference
between a script and a coffee break. It is also safe *here specifically*: `A` is SPD and
only grows, so the denominator is bounded below by 1 and there is no cancellation. That
is not true of the downdate, which is why the class never removes an observation. Measured
against an explicit re-inversion after 600 updates, the worst relative disagreement is
**9.0 × 10⁻¹⁶** — [`tests/test_bandits.py`](tests/test_bandits.py).

Thompson sampling carries the identical `A⁻¹` and draws `θ̃_a ~ N(θ̂_a, v²A_a⁻¹)`; its
posterior standard deviation in direction `x` is `v√(xᵀA⁻¹x)`, LinUCB's width up to the
constant. Optimism replaces the sample by a deterministic quantile. The test suite asserts
the two classes agree on that quantity to ten decimal places.
→ [`src/routers/thompson.py`](src/routers/thompson.py)

### 2. Why the budget collapses to a single number

Over `T` queries, maximise quality subject to a hard compute budget:

```
maximise  Σ_t q_{t,a_t}      subject to   Σ_t c_{t,a_t}  ≤  B
```

a multiple-choice knapsack. Dualise the one coupling constraint with a multiplier `p ≥ 0`:

```
L(p)  =  max_{a_1..a_T}  Σ_t ( q_{t,a_t} − p·c_{t,a_t} )  +  pB
```

The inner maximisation **decouples completely** — with `p` fixed, each query is answered
by whichever arm maximises `q − p·c`, using nothing but that query's own numbers. The
whole budget constraint collapses into one scalar. `p*` has a reading: the exchange rate
between quality and compute at the margin, in units of *probability of success per
simulated GPU-second*. It is the same object as `λ`, except that `λ` is a preference you
assert and `p*` is a price the constraint imposes on you.

`L(p)` is convex with subgradient `B/T − c_t`, so projected online gradient ascent learns
it:

```
p_{t+1}  =  max( 0,  p_t + η ( c_t − B/T ) )
```

Overspending against the per-query allowance raises the price, pushing the greedy rule
toward cheaper arms. Measured on the simulated fleet over 20,000 queries: the
unconstrained LinUCB policy spends **1.4286 × B**, while the single-price router spends
**0.9909 × B** and still holds 0.849 task success against LinUCB's 0.908. Its realized
spend tracks the straight line `(B/T)·t` to within **3.6%** after the transient, and the
learned price settles at `p* ≈ 0.26` probability-of-success per simulated GPU-second
(range 0.18–0.32) — [`results/pareto.csv`](results/pareto.csv), `block=budget_trace`,
which carries both series.

**The gap I close explicitly.** The dual controls spend *in expectation*, but the router
plans against expected cost and is billed realized cost, and a proportional controller
tracking a ramp has a steady-state lag. Driven at the full allowance the realized spend
overshoots **every time** — 1.00037 B to 1.00090 B across ten seeds. So the dual is driven
to `(1 − reserve)·B/T` with `reserve = 0.01`, which lands it at 0.9907 B to 0.9915 B and
never violates the constraint. Both figures are asserted in
[`tests/test_budget.py`](tests/test_budget.py), because "the budget is respected" is only
a claim if the version that misses by 0.06% is also on the record.

**And the structural claim is checked, not cited.** The LP relaxation of a multiple-choice
knapsack has integrality gap at most one item, so the best single-price policy must be
within one query's quality of the exact optimum. `offline_knapsack_dp` solves a small
instance by dynamic programming and `best_single_price` sweeps `p`; over six random
40-query instances the single-price policy captures **0.9941 to 1.0011** of the DP
optimum, a shortfall of at most **0.19 of one item** against the bound's 1.0. (Ratios
slightly above 1 are expected and not an error: the DP rounds costs *up* onto its budget
grid, so its value is a marginal underestimate of the true optimum.)
→ [`src/routers/budgeted.py`](src/routers/budgeted.py)

### 3. The escalation rule, and the guarantee it carries

"Escalate if confidence < 0.7" is arbitrary, is not calibrated, and is not comparable
between a 3B and a 32B model, which are confident about entirely different things. Split
conformal replaces it with a threshold computed from held-out data.

Take a calibration set `(X_i, Y_i)_{i≤n}` and a fresh test point, and assume only that the
`n+1` pairs are **exchangeable**. With nonconformity score `s(x,y) = 1 − p̂(y|x)`, the
scores are exchangeable too, so the rank of `s_{n+1}` among all `n+1` of them is uniform
on `{1,…,n+1}`. Writing `s_(k)` for the `k`-th smallest calibration score,

```
P( s_{n+1} ≤ s_(k) )  =  k / (n+1)
```

Take the smallest `k` making this at least `1 − α`, namely `k = ⌈(n+1)(1−α)⌉`, set
`q̂ = s_(k)` — the `⌈(n+1)(1−α)⌉/n` empirical quantile — and define

```
C(x)  =  { y : s(x,y) ≤ q̂ }
```

Then `Y ∈ C(X)` exactly when `s_{n+1} ≤ q̂`, giving the finite-sample, distribution-free
bound `P(Y ∈ C(X)) ≥ 1 − α`, with matching upper bound `1 − α + 1/(n+1)` under continuous
scores. Deferral rule: **accept iff `|C(x)| = 1`**, else escalate.
→ [`src/conformal/calibrate.py`](src/conformal/calibrate.py)

Two implementation points that are easy to get wrong and are handled rather than hidden:
`q̂` must be an *order statistic*, so `np.quantile`'s default interpolation quietly voids
the bound; and `k > n` whenever `n < 1/α − 1`, at which point no finite threshold is
justified and the honest output is the full label set, flagged.

**Measured, simulated fleet** — mean over 20 random calibration/test splits, `n_cal` =
4000, `n_test` = 36,000 ([`results/coverage.csv`](results/coverage.csv)):

| target `1−α` | small arm | mid arm | large arm | mean \|C\| (large) |
|---|---|---|---|---|
| 0.99 | 0.9903 | 0.9901 | 0.9904 | 3.66 |
| 0.95 | 0.9489 | 0.9509 | 0.9509 | 1.17 |
| 0.90 | 0.9009 | 0.9010 | 0.8989 | 1.01 |
| 0.80 | 0.7977 | 0.7995 | 0.8019 | 0.84 |

On the diagonal to three decimals, for arms whose accuracies are 0.70, 0.87 and 0.98 —
which is the point of a distribution-free guarantee: it does not care how good the model
is. (Mean set size below 1 is not a bug: at `α = 0.2` the large arm returns an **empty**
set 18.6% of the time, which is the calibrated statement "no candidate answer is
consistent with this model's belief at this level". The cascade treats it as escalate.)

**And it breaks when the premise does.** Calibrate on an easy workload, test on a hard
one, and coverage collapses — at target 0.90, to **0.483 / 0.600 / 0.690**. Nothing in the
code can detect this; exchangeability is an assumption about the world.
BRIEF.md is explicit that diagnosing this is a result rather than a failure, and it is why
the premise is restated every time the guarantee is.

**The honest limitation, volunteered.** The bound is *marginal*. It promises nothing
conditional on anything, and for a router that is exactly the wrong way round. At
`α = 0.10` the small arm's coverage decomposes by difficulty tercile as

```
easy 0.978      medium 0.919      hard 0.805
```

90% marginal coverage that is 80% on the hard third is a system that is confidently wrong
precisely when it matters. Measuring the decomposition is not a criticism of the method;
it is what the method says.

### 4. Composing the guarantee across tiers

Let `i*(x)` be the tier that answers. A miss by the cascade is a miss by *some* tier:

```
{ Y ∉ C_{i*}(X) }  ⊆  ∪_i { Y ∉ C_i(X) }
```

so for any split with `Σ αᵢ ≤ α`, the union bound gives
`P(Y ∉ C_{i*}(X)) ≤ Σ αᵢ ≤ α`. End-to-end distribution-free coverage for a multi-tier
system, in finite samples. **The premise that makes this legal:** every tier must be
calibrated on an i.i.d. draw from the full query distribution, not on the escalated
stream — which is a selected subpopulation and not exchangeable with the marginal.
→ [`src/conformal/cascade.py`](src/conformal/cascade.py)

**Where it is loose, and by how much.** At `α = 0.2` on the nominal workload the bound
allows 0.200 miscoverage and the cascade realises **0.063** — a slack of 0.137, i.e. the
bound is 3.2× conservative. Two distinct reasons, with different fixes:

1. **The miss events are strongly positively correlated.** A genuinely hard query is
   missed by every tier at once. A union of near-nested events is far smaller than the sum
   of their probabilities.
2. **The second and third tiers are only consulted on the escalated subset.** Their
   miscoverage on the queries that never reach them is charged to the budget and then
   never incurred. This part grows with the first tier's acceptance rate: a cascade that
   accepts 80% of queries at the first tier pays for the other two tiers' full `αᵢ` while
   exposing itself to only 20% of it.

Tightening (1) needs joint calibration over the tiers' scores, which costs the
distribution-free property because the escalation event depends on those same scores.
Tightening (2) needs per-tier calibration on the escalated subpopulation, which is not
exchangeable with the marginal — so it buys a *different* guarantee, not a tighter version
of this one. Which of the two you would spend is a design decision, not a detail.

**A structural condition that falls out, and is visible in the data.** A tier can only
ever emit a singleton — and so can only ever answer — when its allotted `αᵢ` exceeds its
own error rate. Below that, the `(1−αᵢ)` calibration quantile lands inside the mass of
scores from its own wrong answers, `q̂` goes high, and every set comes back wide. Measured
at `α = 0.2`, simulated fleet:

| Workload | split | tier error rates | `αᵢ` | acceptance per tier |
|---|---|---|---|---|
| nominal | equal | 0.298 / 0.133 / 0.020 | 0.067 each | 0.001 / 0.039 / **0.960** |
| easy | equal | 0.075 / 0.021 / 0.003 | 0.067 each | 0.248 / **0.530** / 0.222 |
| easy | front-loaded | 0.075 / 0.021 / 0.003 | 0.14 / 0.04 / 0.02 | **0.575** / 0.257 / 0.168 |

Only tiers with `αᵢ > errorᵢ` answer. On the nominal workload that is the large arm alone,
so the cascade degenerates into "consult all three, use the last" — and costs **4.33
GPU-s/query against 2.81 for calling the large arm directly**. That is a genuinely bad
result for the cascade, it is in the Pareto table above, and it is the consequence of the
condition rather than a bug. Front-loading the budget onto the tier that sees every query
is the fix when the arithmetic allows it: on the easy workload it moves *first-tier*
acceptance from 0.248 to 0.575 and the cost from 1.99 to 1.48 simulated GPU-s/query,
below always-largest's 2.81.

### 5. Multi-step compounding, and which way the correlation points

`n` independent steps at per-step success `p` give `pⁿ`. That is the quantitative argument
for why routing matters more in an agent than in a chatbot — but quoting `pⁿ` and then
reporting `pⁿ` as the measured end-to-end rate measures nothing. So
[`src/agent/loop.py`](src/agent/loop.py) runs actual episodes: a real calculator (AST
whitelist, not `eval`), real file reads, real substring search, with only the *decision
about whether the model emitted a correct tool call* coming from the simulated fleet.
An episode succeeds only if every step's real output matches ground truth.

With a shared per-episode difficulty, the end-to-end rate is `E_d[p(d)ⁿ]`, and `q ↦ qⁿ` is
convex, so Jensen gives

```
E_d[ p(d)ⁿ ]  ≥  ( E_d[ p(d) ] )ⁿ  =  p̄ⁿ
```

**Correlated failures make an agent do better than the independence formula predicts, not
worse** — failures bunch into a minority of hard episodes instead of spreading evenly.
Measured over 1,500 episodes at `n = 12`, simulated fleet
([`results/compounding.csv`](results/compounding.csv)):

| Policy | regime | per-step `p` | predicted `pⁿ` | measured | ratio |
|---|---|---|---|---|---|
| always-small | **independent** | 0.7535 | 0.0335 | 0.0327 | **0.98** |
| always-small | correlated | 0.7448 | 0.0291 | 0.1993 | **6.84** |
| LinUCB | **independent** | 0.9203 | 0.3693 | 0.3773 | **1.02** |
| LinUCB | correlated | 0.9173 | 0.3548 | 0.4233 | 1.19 |
| always-large | **independent** | 0.9841 | 0.8251 | 0.8233 | 1.00 |
| always-large | correlated | 0.9804 | 0.7885 | 0.8147 | 1.03 |

The independent rows are the control that makes the rest credible: draw each step's
difficulty afresh and the measurement lands on `pⁿ` to within 2%, so the loop is measuring
what the formula predicts. Switch the correlation on and the weak model's end-to-end rate
is **6.8× the `pⁿ` prediction**. The gap is largest exactly where `p` varies most across
episodes, and it shrinks to nothing for an arm that is uniformly good. Anyone quoting
`pⁿ` for a weak model on correlated tasks is off by most of an order of magnitude.

---

## The limitation I volunteer first

**The regret bound is over the reward I actually optimise, not the reward I care about.**
A biased reward surrogate does not weaken the bound — it removes its premise. The policy
still converges, at the rate the theory promises, to the maximiser of the wrong objective,
and nothing in its own diagnostics can tell it so. This is the sharpest weakness in the
project, so [`eval/surrogate_bias.py`](eval/surrogate_bias.py) produces it on purpose and
measures both halves.

An LLM judge scoring its own fleet tends to reward the fluent, longer, more
authoritative-sounding answer — systematically the bigger model's, correct or not. Model
that as capability-proportional partial credit, `q̃ = success + b·w_a` with
`w = (0, 0.386, 1)`. The bandit learns from `q̃ − λc` and is scored on the truth. Simulated
fleet, 40,000 queries × 3 seeds, fitted over `t ∈ [2000, 40000]`:

| bias `b` | exponent, regret vs the **surrogate** optimum | exponent, regret vs the **true** optimum | true reward | true success | cost | share to largest arm |
|---|---|---|---|---|---|---|
| 0.00 | 0.846 | 0.846 | 0.7504 | 0.9066 | 1.302 | 0.167 |
| 0.05 | 0.821 | 0.812 | 0.7494 | 0.9191 | 1.414 | 0.219 |
| 0.10 | 0.762 | 0.776 | 0.7451 | 0.9317 | 1.555 | 0.288 |
| 0.20 | 0.805 | 0.908 | 0.7183 | 0.9583 | 1.999 | 0.536 |
| **0.35** | **0.802** (R² 0.999) | **1.052** | **0.6527** | 0.9756 | 2.691 | **0.932** |

At `b = 0.35` the policy's own regret curve is a clean sublinear power law with R² 0.999 —
it looks *healthy* — while its regret against the objective that matters is growing
**linearly**. It has routed 93% of traffic to the largest arm, doubled its compute bill,
raised task success to 0.976, and destroyed 13% of true reward. Every internal diagnostic
says it is working.

There is no threshold at which this switches on. The damage is continuous in `b`, which is
worse than a threshold, because a small bias is invisible rather than absent. The only
defences are to measure the surrogate's agreement with ground truth on a human-labelled
sample — **which I cannot do here, and which is listed in [STATUS.md](STATUS.md) as not
obtainable** — and to bound how far the optimum can move under a bias of a given size,
which is the last column.

**Second: the fleet is simulated, and I chose its parameters.** The mechanisms above are
measured; the magnitudes are properties of a simulator I tuned to make routing
non-degenerate. That tuning is stated in the simulator's source and asserted in the tests
(the oracle must use every arm on more than 10% of queries), but it is still a choice I
made, not a fact about the world. Nothing here should be read as a claim about what any
real model would do.

**Third: the linear reward model is misspecified on the fleet, and it is why the bandit
loses.** The true response curve is a logistic in a latent difficulty the router sees only
through a noisy classifier. `features.py` adds a quadratic term in that score as the best
a linear model can do against a sigmoid, and the residual is a floor. Two pieces of
evidence rather than an assertion:

- LinUCB's regret exponent on the fleet is 0.858, not ½ — and the ½ is labelled
  throughout as a statement about the well-specified *synthetic* bandit, never about the
  fleet.
- More tellingly, that exponent **is not stable in the horizon** — it climbs from
  0.786 to 0.890 as the fit window is extended from
  `[2000, 40000]` to `[2000, 1,200,000]` (the table in
  [§3 above](#3-the-simple-baseline-wins-at-20000-queries-and-is-never-behind-at-12-million)). A
  well-specified learner's exponent settles; a misspecified one's climbs toward 1,
  because the approximation error contributes a constant per-query regret that no amount
  of data removes. That climb is the misspecification floor, measured directly rather
  than argued for, and it is why LinUCB never catches the frozen classifier.

The fix is not a bigger bonus or more exploration — it is a feature map that can represent
the response curve, or a non-linear reward model, and neither is a bandit question.

**Fourth: LinGreedy reaching the same exponent as LinUCB on the synthetic bandit was not
the plan.** Contexts there are drawn i.i.d. from the unit sphere, so they explore the
parameter space *for* the policy and a greedy learner is self-exploring; greedy's linear
worst case needs instances that starve it of that diversity. The exploration bonus does
earn its keep on the fleet, where the model is misspecified — LinGreedy's mean reward
there is 0.652 against LinUCB's 0.748, and it collapses onto the small arm 80% of the
time. Two measurements, one conclusion: exploration pays when the model is *wrong*, not
merely when the problem is stochastic.

---

## Running it

No installation, no downloads, no network. numpy and PyYAML are the only requirements.

```bash
make test        # 63 tests, ~61 s
make results     # regenerates every CSV in results/  (~22 min; eval/regret.py is 19 of it)
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

The suite is deliberately not fast. The headline regret test runs the real minimax
protocol — the same `build_envelope` that `eval/regret.py` calls, at `T = 16,000` over 8
gap scales and 5 seeds — and it is by a wide margin the largest single test in the suite.
The seed count is the expensive part and it is not negotiable: at 2 seeds the fitted
exponent moves by ±0.1 depending on which instance family is drawn, at 5 it moves by
±0.025. Shrinking `T` or the seeds until the suite was quick would have been shrinking
the claim.

Two assertions carry the project, marked in the source with `=== THE TEST THAT MATTERS ===`:

- **The regret exponent.** LinUCB's minimax envelope exponent is ½ and the random
  policy's is 1, under an identical protocol, with the log–log fit's R² checked first
  because a slope fitted to a curve that is not straight is not a slope. The test also
  asserts that the maximising gap is *interior* to the family — a supremum attained on the
  grid edge would mean the family was truncating the answer — and that it shrinks with
  `T`. → [`tests/test_bandits.py`](tests/test_bandits.py)
- **Coverage.** Empirical coverage meets `1 − α` across the whole sweep, for all three
  arms, on held-out data, averaged over 12 splits — *and* the matching upper bound
  `1 − α + 1/(n+1)`, so that a method returning the full label set every time would fail.
  The cascade's composed miscoverage never exceeds `Σ αᵢ`.
  → [`tests/test_conformal.py`](tests/test_conformal.py)

Alongside them: the budgeted router stays under `B` while the unconstrained one does not;
the single-price rule matches an offline knapsack DP within the one-item integrality gap;
Sherman–Morrison agrees with explicit re-inversion to 9 × 10⁻¹⁶; no policy ever beats the
expected-reward oracle; conformal coverage *breaks* under deliberate distribution shift;
the calculator refuses `__import__('os').system(...)`; and the agent loop loses no episode
when handed a hypothetical perfect arm.

### A note on the two oracles

They are different objects and the README uses both, so they are named:

- **Oracle (expected-reward)** — `argmax_a E[r | x_t, a]`, knowing the latent difficulty.
  This is the regret benchmark. No policy can beat it in expectation, and the test suite
  asserts exactly that, on expected reward rather than on success rate.
- **Oracle (cheapest sufficient)** — `argmax_a (success_{t,a} − λ·c_{t,a})` using
  *realized* outcomes; on this fleet it agrees with "the cheapest arm that actually
  succeeded" on 99.99% of queries. It reads the answer key, so it is not achievable and it
  has *positive* regret under the expected-reward accounting — which looks paradoxical
  until you notice it is exploiting realized noise that no policy can anticipate. It
  exists to bound the Pareto table from above.

### Reading the tables

`Peak mem` is the **maximum** resident memory over the arms a policy actually touches —
what you must provision. A router sending 2% of queries to the 32B arm still needs it
resident, so its provisioning cost is that arm's, not its average; the two differ by more
than 4× here, and conflating them is how routing results get oversold.
`Escalation` means "share of queries not sent to the cheapest arm" for a bandit router and
"share not answered by the first tier" for the cascade; it is blank for fixed arms and
oracles. The cascade row's `Task success` is the answering tier's argmax correctness — the
answer you would act on — which is a different notion of correct from set coverage, and
the two live in different tables on purpose.

## Layout

```
src/fleet/simulator.py     the instrument: ladder, correlated success, cost, both oracles
src/fleet/client.py        the same interface over a real server; raises, does not mock
src/cost.py                one cost interface; GPU-seconds now, $/token by substitution
src/features.py            the serving-time context, with the availability audit
src/routers/               baselines · linucb · thompson · budgeted
src/conformal/             calibrate (split conformal) · cascade (deferral + composition)
src/agent/                 loop · tools (calculator, file read, text search)
eval/                      the five experiments; each writes one results/*.csv
analysis/plot_results.py   the only file that imports matplotlib
serve/fleet.yaml           what a real deployment would serve; nothing here runs it
```

[BRIEF.md](BRIEF.md)'s suggested layout names `eval/` for the number-producing scripts, so
this repo follows it there rather than the portfolio's default `analysis/`; `make results`
runs `eval/`, `make plots` runs `analysis/`.

## References

- Lattimore, T. & Szepesvári, C. [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf)
  (Cambridge UP; free PDF from the author) — the source for every regret bound used here:
  the self-normalised tail bound and elliptical potential argument behind LinUCB's
  `Õ(d√T)`, and the corresponding rate for linear Thompson sampling.
- Angelopoulos, A. & Bates, S. [*A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification*](https://arxiv.org/abs/2107.07511)
  (arXiv:2107.07511) — the split-conformal construction and the `⌈(n+1)(1−α)⌉/n` quantile.
  Reference code: [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction).
- [Bandit Formulations of Model Routing](https://www.tmls.nyc/research/bandit-model-routing) —
  the knapsack / single-price structural result implemented in `src/routers/budgeted.py`.
- [Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference](https://arxiv.org/html/2607.25018v2) —
  coverage composition across tiers.
- [Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing](https://arxiv.org/html/2607.09015v1) —
  directly on the surrogate-reward bias measured above.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) —
  where the arms of a real fleet would be chosen, at build time.

Full brief: [BRIEF.md](BRIEF.md). Honest scope: [STATUS.md](STATUS.md).
