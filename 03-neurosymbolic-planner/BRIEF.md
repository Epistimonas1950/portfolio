# 03 · Neurosymbolic Planner: the LLM Proposes, Exact Search Guarantees

> The only framing in which A* and alpha–beta stop being an undergraduate exercise.

| | |
|---|---|
| **Effort** | 2–3 weeks |
| **Prerequisites** | None — independent of `01`/`02`, can run in parallel |
| **Feeds** | Tier 3 (this is the flagship you wrap as a deployed service) |
| **Math** | Admissibility, consistency, heuristic dominance, branching factor, LP relaxation |
| **Status** | ☐ not started |

---

## The problem

Everyone knows LLMs plan badly over long horizons. The usual response is a bigger model or more
chain-of-thought — both of which degrade gracefully into being wrong with confidence.

The correct response is architectural: **let the LLM do the part it is good at (translating messy
natural language into a formal problem instance) and let exact search do the part it is good at
(returning a provably optimal plan).** Then let the LLM narrate the result back.

The value of this project is that it inverts the usual demo. Instead of "look, the LLM did
something impressive most of the time," you get "this system's plan is optimal by construction,
and here is the one component that can fail, measured separately."

## Pick a domain with real structure

Avoid blocksworld — it reads as a textbook exercise. Good choices:

- **Multi-stop delivery with time windows** (a routing problem with real constraints)
- **Machine-shop / job-shop scheduling** with setup costs
- **Warehouse order picking** with aisle geometry

Any of these gives you a natural NL front end ("get these six parcels delivered before 3pm, the
van holds four") and a genuinely hard combinatorial core.

## The mathematics

**1. Admissibility.** Your heuristic `h(n)` must never overestimate the true cost-to-go `h*(n)`,
or A* loses its optimality guarantee. State your `h` and prove the bound.

**2. Consistency.** The stronger condition `h(n) ≤ c(n, n') + h(n')` for every edge. Prove that
consistency implies `f` is non-decreasing along any path, and therefore that a node is never
re-expanded — which is why the closed set is safe. Consistency vs. admissibility is a
standard interview question and most candidates only half-know the difference.

**3. Heuristic dominance.** If `h₂(n) ≥ h₁(n)` everywhere and both are admissible, `h₂` expands no
more nodes than `h₁`. This turns "make the heuristic better" into a measurable, provable claim
rather than a vibe.

**4. Where to get a good heuristic.** Three sources, all defensible:
   - **Relaxed problem** — drop a constraint and solve exactly (e.g. ignore capacity → MST/shortest
     path bound).
   - **LP relaxation** — solve the linear relaxation of the integer program; the optimum is an
     admissible bound by construction.
   - **Pattern database** — precompute exact costs for projected sub-states, store, look up.

**5. Adversarial variant.** If your domain has a competing agent, add alpha–beta with iterative
deepening. Derive why perfect move ordering gives `O(b^(d/2))` instead of `O(b^d)` — the square
root is the entire point of the algorithm — then measure your *actual* effective branching factor
and show how much move ordering and transposition tables bought you.

## What to build

- [ ] Formal problem representation (state, transitions, cost model) with a validator
- [ ] LLM translation layer: NL → JSON instance, using structured/constrained output
- [ ] A* with a pluggable heuristic + IDA* for the memory-bound case
- [ ] At least two heuristics so you can demonstrate dominance empirically
- [ ] Alpha–beta with iterative deepening + transposition table (if adversarial)
- [ ] Instrumentation: node expansions, effective branching factor, wall clock, optimality check
- [ ] Baseline: pure-LLM chain-of-thought planner on the identical instances
- [ ] **Separate benchmark for the translation step alone** (see limitations)
- [ ] LLM explanation layer: plan → natural language
- [ ] Gradio demo — type a request, watch the search, read the explanation

## How it's measured

Two independent evaluations. Keeping them separate is the mark of the project.

**A. The search (should be perfect):**

| Horizon | Instances | Optimal found | Node expansions | Eff. branching factor | Time |
|---|---|---|---|---|---|
| 5 steps | | 100% | | | |
| 10 steps | | 100% | | | |
| 20 steps | | 100% | | | |

**B. Search vs. pure LLM (the headline plot):**

| Horizon | A* optimality | LLM optimality | LLM feasibility |
|---|---|---|---|
| 5 | 100% | | |
| 10 | 100% | | |
| 20 | 100% | | |

The LLM curve falls off; yours is a flat line at 100%. That plot is the project.

**C. Translation accuracy (the real system error rate):** N hand-written NL requests, measured
for exact-match instance extraction, with a taxonomy of failure modes.

## The limitation you volunteer first

**The LLM-to-formal-spec step is the entire failure surface, and no amount of better heuristic
fixes it.** The end-to-end system is exactly as reliable as the translation. Instrument it as its
own benchmark, publish its accuracy separately, and be able to say "the system is 100% optimal
*given a correct parse*, and parses correctly 9X% of the time — so here's where I'd spend the
next engineering month."

That sentence is worth more than the whole demo.

## Interview claim

> I know exactly which part of this system can be wrong. The search is proven optimal, so every
> failure is a translation failure — and I measure that separately, so I can tell you the system's
> true error rate and where to spend engineering effort.

## Stack

Python for orchestration · Rust or C++ for the search core if you want the performance story ·
a **local open-weight model** with structured output — project `08` builds exactly that fleet, so
this costs nothing to run or demo · Gradio for the demo

## Suggested repo layout

```
neurosymbolic-planner/
  README.md              <- the optimality-vs-horizon plot at the top
  src/
    domain.py            state, transitions, cost, validator
    translate.py         NL -> instance (LLM)
    search/
      astar.py
      idastar.py
      alphabeta.py
      heuristics.py      relaxed | lp | pdb
    explain.py           plan -> NL
  bench/
    vs_llm.py
    translation_accuracy.py
  app/
    demo.py              Gradio
  results/
    optimality_vs_horizon.png
    expansions.csv
```

## References

- Hart, Nilsson & Raphael (1968). [*A Formal Basis for the Heuristic Determination of Minimum Cost
  Paths*](https://doi.org/10.1109/TSSC.1968.300136), IEEE Trans. SSC 4(2):100–107 — the
  admissibility result, worth reading in the original.
  [Free PDF](https://people.stfx.ca/jdelamer/courses/csci-564/_downloads/b2220c66675ddde471ca1795147b8e86/A_Formal_Basis_for_the_Heuristic_Determination_of_Minimum_Cost_Paths.pdf).
- Korf, R. (1985). [*Depth-first Iterative-Deepening: An Optimal Admissible Tree
  Search*](https://doi.org/10.1016/0004-3702%2885%2990084-0), Artificial Intelligence
  27(1):97–109 — for IDA*.
- Knuth, D. & Moore, R. (1975). [*An Analysis of Alpha-Beta
  Pruning*](https://doi.org/10.1016/0004-3702%2875%2990019-3), Artificial Intelligence
  6(4):293–326 — the `b^(d/2)` result.
- Pearl, J. *Heuristics: Intelligent Search Strategies for Computer Problem Solving* (1984) —
  book, no free copy. Dominance and the theory of heuristic search.
