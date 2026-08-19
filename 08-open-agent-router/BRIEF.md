# 08 · A Cost-Optimal Agent Over a Fleet of Free Open Models

> An agent that routes every call to the cheapest open-weight model that can handle it —
> with a regret bound on the routing policy and a coverage guarantee on the escalation rule.

| | |
|---|---|
| **Effort** | 3 weeks |
| **Prerequisites** | None. **Build this second, before `02`**, if the job description says "agents" |
| **Feeds** | `03` (supplies its local model, replacing the paid API) |
| **Cost** | Zero — every model is open-weight and runs locally |
| **Math** | Contextual and budgeted bandits, √T regret, split conformal prediction, coverage composition |
| **Status** | ☐ not started |

---

## Why this one is worth a flagship slot

Two facts, both from the current hiring signal:

1. **Every AI job description now says "agents."** An agent project is the one thing on this list a
   hiring manager is actively looking for by name.
2. **Nearly every agent project is a framework demo.** Wire up a framework, give it three tools,
   show it booking a flight. There is no measurement, no baseline, and no mathematics.

The gap between those two facts is your opening. An agent has to decide *which model to call*, and
that decision is a sequential decision problem under uncertainty with a cost constraint — which is
a solved problem in the bandit literature, with theorems. Almost nobody building agents knows
that.

And because every model here is open-weight and local, **the whole project costs nothing to run**
and produces the numbers a company cares about most: quality at a given compute bill.

## What it is

An agent (tool-using, multi-step) whose every LLM call is dispatched by a learned router to one of
several open-weight models running on your own hardware. Small models answer easy calls; the
router escalates only when it must, and the escalation rule carries a formal guarantee.

```
                 ┌─────────── router (contextual bandit) ────────────┐
   task ──► agent│  features(query) ──► arm choice ──► model call    │──► tool ──► ...
                 └──────────────┬───────────────────────────────────┘
                                │  conformal set too large?
                                └──► escalate to the next tier up
```

## Choosing the fleet

**Pick the arms at build time from the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), not from this document.** Specific model names age in
months; the selection criterion does not. What you need is a **capability ladder wide enough to
make routing non-trivial** — if all arms are equally good, there is nothing to learn.

| Tier | Size | Role |
|---|---|---|
| Small | 1–4B | Handles the majority of easy calls; near-free |
| Mid | 7–8B | The workhorse |
| Large | ~30B, quantized | Escalation target for hard calls |

As of writing, the Qwen3 family is the default local answer, with Llama 3.x, Mistral/Devstral and
GLM as alternatives — but check the leaderboard and pick current models with genuine tool-calling
support. Serve them with `llama.cpp`, Ollama, or vLLM.

**Two hardware configurations, pick one and say which:**

- **Config A (GPU, ≥24 GB VRAM):** the full three-tier ladder including the quantized 30B-class arm.
- **Config B (CPU only):** three arms in the 1–8B range via `llama.cpp`. The Pareto front is
  compressed but **still measurable** — every result in this brief is reproducible on CPU, just
  slower. Do not skip the project for lack of a GPU; state the configuration and move on.

## What "cost" means here

Everything is local and open-weight, so there are **no dollars**. Define cost as **measured
wall-clock seconds and peak VRAM per call** — that is the real quantity a company pays for, since
GPU-seconds are the bill. Keep the cost model behind one interface so a `$/token` table for a
hosted fleet swaps in without touching the policy. Say this in the README's first paragraph, or a
reviewer will reasonably ask what "cost-optimal" means when all the models are free.

## The mathematics

### 1. Routing is a budgeted contextual bandit

At step `t` a query arrives with feature vector `x_t` (length, task type, tool-call depth,
embedding, difficulty-classifier score). You choose an arm `a_t ∈ {1..K}` (a model), observe
reward `r_t` and pay cost `c_t`. Reward combines quality and price:

```
r_t  =  quality_t  −  λ · cost_t
```

**Unconstrained case.** With a linear reward model `E[r | x, a] = θ_aᵀ x`, LinUCB and Thompson
sampling both achieve

```
Regret(T)  =  Õ( d √T )
```

Implement both. Then **verify the exponent empirically**: plot cumulative regret against `T` on
log–log axes and read the slope. It should be ≈ 0.5. This is the same move as the convergence-order
plot in project `04` — a theoretical rate, confirmed by measurement — and it is the single most
convincing artifact in the project.

**Budgeted case.** Add a hard constraint: total cost over `T` queries must not exceed `B`. This is
a knapsack contextual bandit, and the structural result is worth knowing — **the optimal policy is
a single-price (threshold) rule**: there exists a price `p*` on cost such that acting greedily on
`quality − p* · cost` is optimal, and `p*` can be learned online with regret still `O(√T)`.
Implement it as a dual-variable update on the budget constraint and show the realized spend
tracking the budget line.

### 2. Escalation is a conformal cascade

The naive deferral rule is "escalate if the small model's confidence is below 0.7." That threshold
is arbitrary, it is not calibrated, and it gives you no guarantee. Replace it with **split
conformal prediction**.

Hold out a calibration set. Define a nonconformity score `s(x, y)` (negative log-likelihood of the
emitted answer, or a token-margin score). Let `q_hat` be the `⌈(n+1)(1−α)⌉/n` empirical quantile of
calibration scores. The prediction set

```
C(x)  =  { y : s(x, y) ≤ q_hat }
```

satisfies, for any distribution, with no assumption but exchangeability, and in **finite samples**:

```
P( Y ∈ C(X) )  ≥  1 − α
```

**Deferral rule:** accept the tier's answer iff `|C(x)| = 1`; otherwise escalate. The set size is
now a calibrated measure of "this model isn't sure," not a hand-tuned number.

**Composition across tiers.** A cascade needs the guarantee to survive stacking. Split the error
budget across tiers, `Σ αᵢ ≤ α`, and get an end-to-end bound by union bound — then note where
that is conservative and what you would need for a tighter one. Being able to say *"the union
bound is loose here, and here is the empirical gap"* is exactly the kind of remark that reads as
a mathematician rather than a library user.

**Then verify it.** Sweep `α ∈ {0.01 … 0.2}`, plot target coverage against empirical coverage on a
held-out set, and show the diagonal. If it doesn't sit on the diagonal, your exchangeability
assumption is violated — and diagnosing *why* (distribution shift between calibration and test) is
a result, not a failure.

### 3. Multi-step error compounding

Agents are multi-step, so per-step reliability matters superlinearly. If each of `n` steps succeeds
independently with probability `p`, the task succeeds with `pⁿ`: at `p = 0.95`, a 10-step task
succeeds 60% of the time; at `p = 0.90`, 35%.

This is the quantitative argument for *why* routing matters in an agent and not just in a chatbot —
a cheap model that is 4 points worse per call is catastrophically worse over a horizon. **Measure
your own per-step success rate, predict the end-to-end rate from it, and compare to the measured
end-to-end rate.** The gap tells you how correlated your step failures are, which is a genuinely
interesting quantity and one nobody else will have.

## The reward signal is the failure surface

Your bandit learns from a per-query correctness signal that you will not have at serving time.
State which one you are using, explicitly:

| Source | Honest description |
|---|---|
| Held-out labeled set | Clean; offline evaluation only; cannot learn online |
| LLM-as-judge | Usable online — **must report agreement rate with human labels on a sample** |
| Surrogate reward | Cheapest; biased |

**Volunteer this limitation before you are asked:** the regret bound is over *the reward you
actually optimize*, not the reward you care about. A biased surrogate breaks the bound's premise
entirely — the policy converges correctly to the wrong objective. This is the sharpest weakness in
the project, and an interviewer with ML experience will look for it. Getting there first is worth
more than the result.

## The experiment that decides whether the maths earned its place

**With 4–6 arms and good context features, a threshold on a cheap difficulty classifier often
matches a contextual bandit.** Make that a first-class row in the results table, not a footnote.

The project's real claim is *"here is where the bandit earns its complexity, measured."* If the
simple threshold wins on your workload, **publish that** — it is a stronger result than a fancier
method with no baseline, and it is the same standard applied everywhere else in this portfolio.

## How it's measured

**A. Cost–quality Pareto** (the headline plot — routers as points, frontier as a line):

| Policy | Task success | GPU-s / query | Peak VRAM | Escalation rate | Cost @ matched accuracy |
|---|---|---|---|---|---|
| Always smallest | | | | — | |
| Always largest | | | | — | |
| Random arm | | | | — | |
| **Difficulty-threshold classifier** | | | | | |
| LinUCB | | | | | |
| Thompson sampling | | | | | |
| Budgeted (single-price) | | | | | |
| Oracle (cheapest sufficient) | | | | — | — |

**B. Regret:** cumulative regret vs. `T`, log–log, with a √T reference line and the fitted slope.

**C. Conformal coverage:** target `1−α` vs. empirical coverage, plus mean set size and escalation
rate per tier.

**D. Compounding:** measured per-step success `p`, predicted `pⁿ`, measured end-to-end.

**Workload:** use an open agent benchmark (a GAIA subset, or `τ`-bench style tool-use tasks) plus a
self-built suite in one domain you control. Say how many queries — bandit results on 200 queries
are noise; you want thousands, which is affordable precisely because the models are free.

## What to build

- [ ] Serve 3+ open models locally behind one interface (`llama.cpp` / Ollama / vLLM)
- [ ] Per-call instrumentation: wall-clock, tokens, peak VRAM — the cost model
- [ ] Agent loop with 3–5 real tools (search, calculator, file I/O, code execution)
- [ ] Feature extractor for the routing context
- [ ] Baselines: always-small, always-large, random, difficulty-threshold classifier
- [ ] LinUCB and Thompson sampling routers
- [ ] Budgeted single-price router with online dual update
- [ ] Split-conformal calibration + set-size deferral rule
- [ ] Multi-tier coverage composition, with the empirical gap measured
- [ ] Oracle policy (offline, with labels) as the frontier
- [ ] Regret, coverage, Pareto and compounding plots
- [ ] Judge-agreement measurement if you use LLM-as-judge

**Framework:** `smolagents` is the fastest path to a single-agent loop and stays out of your way;
LangGraph gives more control over state if you need branching. Either is fine — **the router is
the project, the framework is plumbing.** Do not let framework choice become the story.

## Interview claim

> Agents call a model on every step, and most systems call the biggest one every time. I route
> per-call with a bandit whose regret I measured at the √T rate the theory predicts, and I escalate
> on a conformal set-size test that gives a distribution-free coverage guarantee instead of a
> hand-tuned confidence threshold. Same task success at N× less compute — and I can show you the
> case where a simple classifier does just as well.

## Stack

`llama.cpp` / Ollama / vLLM · `smolagents` or LangGraph · NumPy/SciPy for the bandit and conformal
machinery (no ML framework needed for either) · open-weight models only — **zero API cost**

## Suggested repo layout

```
open-agent-router/
  README.md              <- Pareto plot + regret slope at the top
  serve/
    fleet.yaml           arms, quantization, serving flags
    client.py            one interface, per-call cost instrumentation
  src/
    features.py          routing context
    routers/
      baselines.py       always-small | always-large | random | threshold
      linucb.py
      thompson.py
      budgeted.py        single-price rule + dual update
    conformal/
      calibrate.py       split conformal, nonconformity scores
      cascade.py         set-size deferral + composition
    agent/
      loop.py            smolagents / LangGraph
      tools.py
  eval/
    workload.py          benchmark subset + your own suite
    regret.py            log-log slope fit
    coverage.py          target vs empirical
    compounding.py       p^n vs measured
  results/
    pareto.png
    regret_loglog.png
    coverage.png
```

## References

**Bandits and routing**
- Lattimore, T. & Szepesvári, C. [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf) (Cambridge UP; **free PDF from the author**, companion site [banditalgs.com](https://banditalgs.com/)) — the standard reference for every regret bound in this project.
- [Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey](https://arxiv.org/pdf/2603.04445)
- [Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing](https://arxiv.org/html/2607.09015v1) — directly on the surrogate-reward bias problem above
- [A New Regret-analysis Framework for Budgeted Multi-Armed Bandits](https://www.jair.org/index.php/jair/article/view/16261)
- [Bandit Formulations of Model Routing](https://www.tmls.nyc/research/bandit-model-routing) — the knapsack/single-price result
- Ong, I. et al. *RouteLLM* — the practical routing baseline everyone knows

**Conformal prediction**
- Angelopoulos, A. & Bates, S. [*A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification*](https://arxiv.org/abs/2107.07511)
  (arXiv:2107.07511) — **start here**; code at
  [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction)
- Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* (Springer) — book, no free
  copy. The underlying theory.
- [Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference](https://arxiv.org/html/2607.25018v2) — coverage composition across tiers
- [UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing](https://arxiv.org/html/2605.18796)
- [PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees](https://arxiv.org/html/2605.18812v1)

**Models and frameworks**
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — pick your arms here, at build time
- [The best open source frameworks for building AI agents in 2026](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks)
- [The Best Open-Source and Open-Weight LLMs to Run Locally](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally)
