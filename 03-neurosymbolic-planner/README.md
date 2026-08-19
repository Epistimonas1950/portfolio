# 03 · Neurosymbolic Planner: the LLM Proposes, Exact Search Guarantees

> A delivery planner in which the language part and the guarantee part are separated on
> purpose: the front end turns English into a formal instance, exact search returns a
> plan that is provably optimal, and the one component that can be wrong is benchmarked
> on its own.

**Status:** search, heuristics, adversarial search, the offline translation backend and
the explanation layer all run and are tested with numpy alone. The language-model
translation path and the Gradio demo are stubs that say so — see [STATUS.md](STATUS.md).

---

## The headline result

**A* returned the exact optimum on every one of 240 runs where the optimum could be
checked** — 60 seeded instances spanning 3 to 7 stops, each solved with four different
admissible heuristics, each answer compared against exhaustive enumeration of every
feasible plan. Costs are whole minutes by construction, so this is exact equality, not
agreement to a tolerance. Regenerate with `make results`;
[`results/optimality.csv`](results/optimality.csv).

| Horizon | Instances | Ground truth | Optimal found | Mean expansions | b* (h2) |
|---|---|---|---|---|---|
| 3 stops | 12 | exhaustive enumeration | **100%** | h0 22.4 → h2 8.2 | 1.487 |
| 4 stops | 12 | exhaustive enumeration | **100%** | h0 66.3 → h2 23.2 | 1.676 |
| 5 stops | 12 | exhaustive enumeration | **100%** | h0 155.8 → h2 52.0 | 1.737 |
| 6 stops | 12 | exhaustive enumeration | **100%** | h0 386.4 → h2 84.7 | 1.674 |
| 7 stops | 12 | exhaustive enumeration | **100%** | h0 831.6 → h2 215.4 | 1.744 |
| 8 stops | 6 | h1-vs-h2 agreement | **100%** | h1 680.5 → h2 283.0 | 1.648 |
| 10 stops | 6 | h1-vs-h2 agreement | **100%** | h1 3 293.0 → h2 1 017.7 | 1.639 |
| 12 stops | 6 | h1-vs-h2 agreement | **100%** | h1 6 928.3 → h2 1 476.8 | 1.550 |
| 16 stops | 5 | not verified | — | h2 20 359.4 | 1.582 |
| 22 stops | 1 | wall probe | — | h2, 1.18 M generated in 27 s | — |
| 24 stops | 1 | wall probe | — | **h2 exceeds a 2 M-node budget after 54 s** | — |

The first five rows are the proved ones: 5 horizons × 12 instances × 4 heuristics = 240
runs, every answer checked against the true optimum. The next three substitute a weaker
check — two admissible heuristics exploring different node sets must return the same
cost. The last three measure the cost of search, not its correctness, and the
`ground_truth` column of the CSV says so; the CSV also carries the 14-, 18- and 20-stop
rows this table leaves out.

The brief asks for a 20-step row. A blank would have been the easy answer; the honest
one is that A* with the strongest heuristic here solves 22 stops to proved optimality
in about half a minute of single-threaded Python and runs out of a 2 M-node budget at 24.
That ceiling is measured rather than guessed. Wall-clock figures are the only numbers in
this README that depend on the machine; every count is exact and reproducible.

**And the second claim, which is the one that is actually interesting:** heuristic
dominance is not a slogan. With `h2 ≥ h1 ≥ h0` pointwise and all three admissible,
expansions fall monotonically, and the gap widens with the horizon
([`results/dominance.csv`](results/dominance.csv), 10 instances per horizon):

| Horizon | h0 (Dijkstra) | h1 (farthest stop) | h2 (spanning tree) | h0/h2 | h1/h2 | hx (inconsistent) |
|---|---|---|---|---|---|---|
| 5 | 1 526 | 852 | **521** | 2.93× | 1.64× | 899, with 36 re-expansions |
| 7 | 8 445 | 5 120 | **1 983** | 4.26× | 2.58× | 4 309, with 460 re-expansions |
| 9 | 34 819 | 17 153 | **5 539** | **6.29×** | **3.10×** | 14 799, with **1 157** re-expansions |

Re-expansions for h0, h1 and h2 are **0 at every horizon and on every instance** — that
is a theorem about consistent heuristics, not a happy accident, and the counter is
asserted equal to zero per instance in the suite. There were also **zero** violations of
the dominance theorem over 60 instance-pairs, and zero total-expansion inversions.

The `hx` column is the control that makes the admissibility-versus-consistency
distinction observable rather than merely stated: an *admissible but deliberately
inconsistent* heuristic returns the identical optimum on every instance and pays for it
with 1 157 re-expansions at 9 stops.

**Alpha–beta, at 8 plies of the adversarial variant, over 6 seeded contests**
([`results/adversarial.csv`](results/adversarial.csv)) — every variant returned the
identical game value, which is asserted, not hoped for:

| Variant | Nodes expanded (6 contests) | vs minimax | b* |
|---|---|---|---|
| minimax | 3 741 180 | 1× | 5.160 |
| alpha–beta | 171 771 | 21.8× | 3.404 |
| alpha–beta + move ordering | 28 833 | 129.8× | 2.720 |
| **alpha–beta + ordering + transposition table** | **13 985** | **267.5×** | **2.465** |

Nominal branching at the root is 9, `sqrt(b) = 3.00`.

**And the number that bounds the whole system**
([`results/translation.csv`](results/translation.csv)) — 31 hand-written requests,
exact-match instance extraction:

| Subset | Requests | Exact match | Wrong and self-flagged | Wrong and silent |
|---|---|---|---|---|
| inside the parser's grammar | 20 | **100.0%** | 0 | 0 |
| deliberately outside it | 11 | 36.4% | 7 | **0** |
| whole corpus | 31 | 77.4% | 7 | **0** |

So: *the plan is optimal given a correct parse, the parse is exact on 100% of in-grammar
requests and 77.4% of a corpus a third of which was written to break it, and on every
request it got wrong it said so first.* The failure that should worry a deployer is the
silent one, and there were none — but there were **4 requests where the parse matched
the gold exactly and information in the request was still thrown away**, which is the
failure mode exact-match scoring cannot see. Both are counted in
[`results/translation_modes.csv`](results/translation_modes.csv).

---

## The problem

Language models plan badly over long horizons, and the standard responses — a bigger
model, more chain-of-thought — degrade into being confidently wrong. The architectural
response is to stop asking one component to do two jobs:

```
   English  ──►  translate  ──►  formal instance  ──►  A*  ──►  optimal plan  ──►  English
                    ▲                                   ▲
              can be wrong                        cannot be wrong
```

Everything to the right of the instance is a theorem. Everything to the left is
English. That split is worth having because it makes the system's error rate *knowable*:
it is exactly the translation error rate, and translation can be benchmarked alone.

**The domain is multi-stop delivery with time windows and vehicle capacity**, not
blocksworld. A state is

```
s = (loc, U, t, cap)
```

— where the van is, the set of parcels still undelivered, the clock, and the units still
on board. Time windows and capacity are hard constraints; the capacity forces trips back
to the depot to reload, so the route is not a single tour. Cost is elapsed time: driving,
plus waiting for a window to open, plus handover, plus reloading. → [`src/domain.py`](src/domain.py)

## The mathematics

### 0. Why this is a shortest-path problem at all, and one honest subtlety

Set `g(s) = t(s) − start_time`. Then every edge cost is `t' − t ≥ 0` and the plan cost is
`g` at the goal, so planning is a least-cost path problem on a finite graph and A* with
an admissible heuristic returns the exact optimum (Hart, Nilsson & Raphael 1968).

Except that it is not *quite* a static shortest-path problem: whether an edge exists
depends on `t`, because a stop's window can close. What rescues it is a monotonicity
argument. Both

```
feasible(p)  ⟺  t + d(loc, p) ≤ latest_p            and
t'           =   max(t + d(loc,p), earliest_p) + service_p
```

are non-decreasing in `t`, so if the same `(loc, U, cap)` is reachable at `t₁ ≤ t₂`, then
by induction on the length of the completion, **every completion feasible from `t₂` is
feasible from `t₁` and costs no more**. That earliest-arrival dominance is what licenses
A* to key its closed set on `(loc, U, cap)` and throw the clock away — without it the key
would have to include `t`, almost nothing would deduplicate, and the heuristic comparison
would be swamped by the transposition rate.

It is also an argument that is easy to believe and easy to get subtly wrong in code,
which is why [`src/search/bruteforce.py`](src/search/bruteforce.py) enumerates every
feasible plan with none of that machinery and the test suite compares the two exactly.

### 1. The lemma, and three admissible heuristics

Every move costs at least its drive,

```
c(s, s')  ≥  d(loc(s), loc(s'))                                              (L)
```

and every completion of `s` is a walk `loc(s) = v₀, …, v_k = depot` whose vertex set is
exactly `U ∪ {loc, depot}` — legal moves go only to an outstanding parcel or the depot.
So any lower bound on the geometric length of such a walk is a lower bound on `h*`.

`d` has to be a metric for the triangle inequality steps below, and that is a coding
decision, not an assumption: [`domain.travel_matrix`](src/domain.py) takes the **ceiling**
of the Euclidean distance, not the nearest integer, because `ceil(a+b) ≤ ceil(a)+ceil(b)`
whereas rounding can break the triangle inequality outright (`1.41, 1.41, 2.83` round to
`1, 1, 3`). The test suite checks the counterexample and checks the generated matrices.

**h0 = 0.** A* degenerates to Dijkstra. The control.

**h1 = maxₚ∈U∪{depot} [ d(loc, p) + d(p, depot) ].** *You still have to reach the most
awkward outstanding stop, and then get home.* Admissible: the walk visits `p` at some
index `j`; by the triangle inequality the prefix costs at least `d(loc,p)` and the suffix
at least `d(p,depot)`. Including `q = depot` in the index set is what supplies the final
drive home once `U` is empty — drop it and `h1` collapses one move early and stops being
consistent.

**h2 = max( h1, MST(U ∪ {loc, depot}) ) + Σₚ∈U serviceₚ + reloads·reload_time.** Three
bounds on three *disjoint* components of the remaining clock:

- *Spanning tree.* Relax away the windows, the capacity, and the requirement that the
  route be a walk; keep only "these vertices must end up connected". The cheapest
  connected structure on a vertex set is its minimum spanning tree, computable exactly by
  Prim in `O(k²)` — relax until the relaxed problem is polynomial, then solve it exactly.
  The walk's edges are a connected spanning subgraph, so they weigh at least the MST.
- *Service.* Every outstanding parcel is handed over exactly once, and that is not drive
  time.
- *Reloads.* With `D` units still to deliver and `cap` on board, `r` reloads make
  `cap + r·capacity` units available, so `r ≥ ⌈(D − cap)/capacity⌉`, each costing at least
  `reload_time` on top of its drive. This is the only term that sees the capacity
  constraint, and it stops A* exploring plans that have stranded the load across town.

**Why h2 takes a max rather than just using the MST.** The MST bound does *not* dominate
h1. Put `loc` and `q` 10 apart with the depot exactly between them: `MST = 10`, but
`h1 = d(loc,q) + d(q,depot) = 15`. The MST is blind to the fact that the walk must
*return*. Taking the max of two admissible heuristics is admissible, and the max of two
consistent heuristics is consistent, so the combination is free.
→ [`src/search/heuristics.py`](src/search/heuristics.py), where each proof sits in the
docstring of the heuristic it belongs to.

### 2. Consistency, and why it is not the same as admissibility

Consistency is the edge condition `h(s) ≤ c(s,s') + h(s')`. It implies `f` is
non-decreasing along any path — `f(s') = g(s) + c + h(s') ≥ g(s) + h(s) = f(s)` — so the
first time A* pops a node, its `g` is already optimal and **no closed node is ever
re-opened**. That is why the closed set is safe.

All three of h0, h1, h2 are consistent, and the proofs go term by term against one edge
cost. For h2 the deficits are: MST term `≤ d(loc,loc')` (add the edge `(loc,loc')` to the
child's tree), service term `= serviceₚ` on a delivery and `0` otherwise, reload term `0`
on a delivery (delivering `p` drops `D` and `cap` by the same `demandₚ`, so `D − cap` does
not move) and at most one `reload_time` on a reload. Summing per move type gives
`d + serviceₚ ≤ c`, `d + reload_time ≤ c`, `d ≤ c` respectively. Consistency does not
compose by adding consistent pieces, so it has to be done this way.

Admissibility alone still buys optimality, provided A* re-opens closed nodes when a
cheaper path appears — and that is a claim you can *measure*.
[`heuristics.InconsistentAdmissibleHeuristic`](src/search/heuristics.py) is h2 suppressed
to 0 on a deterministic pseudo-random half of states: still admissible (0 and h2 are both
lower bounds, and admissibility is pointwise), definitely not consistent (the suppression
ignores edges entirely). The measured consequence, over the 10 seeded instances at each horizon:

| | h0 | h1 | h2 | hx (inconsistent) |
|---|---|---|---|---|
| re-expansions at 9 stops | 0 | 0 | 0 | **1 157** |
| returned the exact optimum | ✔ | ✔ | ✔ | **✔** |

and turning re-opening *off* makes hx return costs that are too high on some instances,
which is asserted in the suite so the demonstration cannot become vacuous.

### 3. Dominance, including the part the theorem does not cover

If `h_b ≥ h_a` everywhere and both are admissible, A* with `h_b` expands no node with
`f < C*` that A* with `h_a` expands (Pearl 1984). Note the strict inequality: nodes
sitting *exactly on* `f = C*` are expanded or not according to tie-breaking, and this
domain has integer costs, so those ties are everywhere.

So the instrumentation carries two counters — `expansions` and
`expansions_below_cstar` — and the test asserts the strict per-instance inequality on the
second, which is the set the theorem actually covers, with the totals asserted in
aggregate. Every heuristic uses the identical tie-break `(f, h, insertion order)`, or the
comparison would be meaningless. On the committed run there were **0 inversions in the
totals over 60 instance-pairs** as well, but that is an empirical fact about these
instances, not a theorem, and the benchmark reports it as such.

### 4. IDA*, and an honest negative result

[`src/search/idastar.py`](src/search/idastar.py) replaces the priority queue with a
sequence of depth-first searches bounded by `f`, each round's bound being the smallest
`f` that overflowed the last. Memory drops to `O(d)`. It returns the identical optimum —
asserted — and on this domain it is much slower: the state graph is nothing but
transpositions (every ordering of the same delivered prefix lands on the same
`(loc, U, cap)`), A* collapses them and IDA* re-explores them in every round. Over the six
7-stop instances it generates 139 060 nodes against A*'s 3 708 -- 37.5x -- while
holding only the current path. That is the finding, and it is the correct
one to report: IDA* is for tree-like state spaces, and this is not one.

### 5. Alpha–beta and the square root

A rival courier works the same map from its own base; the two alternate claiming stops,
each paying its own drive, and the game is zero-sum in the difference of their earnings.
Knuth & Moore (1975): with perfect move ordering alpha–beta visits `O(b^(d/2))` leaves
instead of `O(b^d)` — the same budget buys twice the depth, which is why move ordering is
not a micro-optimisation but the entire algorithm.

The measurement at 8 plies is the table above: `b*` falls 5.160 → 3.404 → 2.720 → 2.465
as pruning, then ordering, then a transposition table are switched on. **What this cannot
show is `b^(d/2)` itself.** That result is asymptotic, for a *uniform* tree, under
*perfect* ordering; here the branching factor is the number of unclaimed stops and so
falls by one every ply, and the ordering is a cheap static heuristic. The honest claim is
the direction and the size of the movement, and the benchmark prints `b` and `sqrt(b)`
next to the measurement so a reader can judge it.

Two implementation points that are mathematical rather than cosmetic. The search returns
the *incremental* value of a position, never the accumulated score, because a
transposition-table entry has to be valid however the search reached the position. And
the alpha–beta window must be shifted by the edge value at every recursion — `v = edge +
child`, so the child is searched in `(α − edge, β − edge)`. Omitting that shift produces
pruning that looks sound and a root value that is wrong by tens of units; it is what the
minimax-equality test caught here. → [`src/search/alphabeta.py`](src/search/alphabeta.py)

### 6. The translation layer, and what its number means

[`src/translate.py`](src/translate.py) puts NL → JSON behind a `TranslationBackend`
interface with two implementations. `RuleBackend` is a deterministic offline parser for a
small grammar fixed in its docstring *before* the corpus was scored: capacity phrases,
gazetteer stop names with the nearest preceding quantity as demand, `before/after/between`
time windows attached to the nearest preceding stop in the same sentence unless they
carry a scope marker, service time, start time. Each stage masks the span it consumed, so
"the van holds four crates" cannot also contribute a demand of four and "before 11:00"
cannot contribute a quantity of eleven. `OllamaBackend` raises `NotImplementedError`
naming the model, the server and the constrained-decoding step it needs.

The parser also flags constructions it can see but cannot represent — an unknown place
name, a relative time, a vague quantity, a conditional, a negation, a coreference, an
ordering constraint. That is why the "wrong and silent" column above is 0: **every
mis-parse announced itself.** An error the system knows about is operationally a
different thing from one it does not.

## Running it

No installation, no downloads, no network, no model. numpy is the only requirement.

```bash
make test        # 42 tests, ~5 s
make results     # regenerates every CSV in results/, ~4 min
make demo        # one English request, end to end
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

`make demo` is the whole thesis in twenty lines of output — request, parse, warnings,
instance, optimal plan, briefing:

```
request: "The van holds three crates. Two crates to the harbour before 11:00, one crate
          to Baker Street between 09:00 and 10:00, and two to Elm Avenue."

1. Drive to the harbour, arriving 08:25. Hand over 2 unit(s); 1 left on board.
   Window closes 11:00 -- 155 min of slack.
2. Drive to baker street, arriving 09:06. ... Window closes 10:00 -- 54 min of slack.
3. Back to the depot at 09:33 to reload -- the van was empty enough that the rest of the
   round could not be carried in one trip.
4. Drive to elm avenue, arriving 10:00. ...
5. Return to the depot, arriving 10:23. Round complete.

Total elapsed: 143 min (115 driving, 0 waiting, 18 handing over, 10 reloading).
No feasible round is faster. This is proved, not estimated: A* with the h2-mst heuristic
settled it in 10 node expansions.
```

## Results

The assertions the project rests on are marked in the source with
`=== THE TEST THAT MATTERS ===`:

- **A*'s cost equals the exhaustive optimum, exactly, on 240 runs, for every admissible
  heuristic** — and the returned plan is re-validated from scratch, so a plan that scores
  well but breaks a window cannot pass. [`tests/test_search.py`](tests/test_search.py)
- **No heuristic overestimates at any reachable state**, not merely at the root: for 678
  live states across nine instances, `h(s)` is compared against `h*(s)` obtained by
  exhaustive enumeration *restarted from s*. This is the hypothesis every other result
  depends on, and the root is the one place a bound is easy to get right by accident —
  h2's reload term `⌈(D − cap)/capacity⌉` is only exercised at partially-delivered states
  with a partial load. Same file.
- **`expansions_below_cstar(h_strong) ≤ expansions(h_weak)`** per instance for all three
  pairs, and the totals in aggregate, with the margins asserted large enough not to be a
  tie. Same file.
- **A consistent heuristic produces exactly 0 re-expansions**, per instance; the
  inconsistent-but-admissible one produces many and stays optimal; and disabling
  re-opening breaks it. Same file.
- **Alpha–beta returns exactly the minimax value** for every variant at every depth while
  expanding strictly fewer nodes. [`tests/test_adversarial.py`](tests/test_adversarial.py)
- **The validator rejects capacity violations and time-window violations separately**, and
  a hand-built three-stop instance whose optimum (26 minutes) was computed on paper is
  asserted as a literal. [`tests/test_domain.py`](tests/test_domain.py)
- **The generated travel matrix is a metric**, checked over all triples — because every
  admissibility proof above depends on it.
- **Every in-grammar request parses exactly right, and every mis-parse is self-flagged.**
  [`tests/test_translate.py`](tests/test_translate.py)

## The limitation I volunteer first

**The translation step is the entire failure surface, and no better heuristic touches
it.** The search is optimal by proof; therefore every end-to-end failure of this system
is a translation failure. That is the good news and the bad news in one sentence, and it
is why translation has its own benchmark, its own CSV and its own failure taxonomy.

**Second, and this one is specific: the corpus is mine.** I wrote all 31 requests and all
31 gold instances, and I wrote the parser. A parser scored against a grammar its own
author also wrote is measuring internal consistency, not the world; 100% on the
in-grammar subset means the grammar is implemented, not that the grammar is enough. The
number that would actually generalise needs traffic I do not have. I wrote the corpus
before the parser and made exactly one general fix afterwards — merging repeated mentions
of the same stop, provoked by an in-grammar request — and left the hard subset alone;
that is the best I can do from inside, and it is not the same as an external test set.

**Third: exact match is the wrong metric and I can show you where.** Four requests parsed
to *exactly* the gold instance while silently discarding a constraint the schema cannot
carry — a precedence requirement ("do the clinic first"), a conditional, a reference to
yesterday's run. Exact-match scoring calls all four successes. They are counted separately
in `results/translation_modes.csv` because a schema that cannot represent what the user
said is a design failure that no parser accuracy number will ever surface.

**Fourth: the pure-LLM baseline curve in BRIEF.md is not here.** The brief's headline plot
is A* optimality against a chain-of-thought planner's on the identical instances. There is
no language model on this machine and no network to reach one, so that curve has not been
measured and the row is empty in the plot script and in STATUS.md. I would rather ship
half a figure than a plausible line no code produced.

**Fifth: 22 stops is not 200.** This is exact search over a state space that is
exponential in the number of parcels, and the wall is measured above. A real dispatcher's
problem is a different algorithm — column generation, or large-neighbourhood search with
a bound — and the honest position is that this repo demonstrates the guarantee, and the
guarantee is what does not scale.

## References

- Hart, P., Nilsson, N. & Raphael, B. (1968). [*A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths*](https://doi.org/10.1109/TSSC.1968.300136), IEEE
  Trans. SSC 4(2):100–107 — the admissibility result.
  [Free PDF](https://people.stfx.ca/jdelamer/courses/csci-564/_downloads/b2220c66675ddde471ca1795147b8e86/A_Formal_Basis_for_the_Heuristic_Determination_of_Minimum_Cost_Paths.pdf).
- Korf, R. (1985). [*Depth-first Iterative-Deepening: An Optimal Admissible Tree
  Search*](https://doi.org/10.1016/0004-3702%2885%2990084-0), Artificial Intelligence
  27(1):97–109 — IDA*.
- Knuth, D. & Moore, R. (1975). [*An Analysis of Alpha-Beta
  Pruning*](https://doi.org/10.1016/0004-3702%2875%2990019-3), Artificial Intelligence
  6(4):293–326 — the `b^(d/2)` result.
- Pearl, J. *Heuristics: Intelligent Search Strategies for Computer Problem Solving*
  (1984) — book, no free copy. Dominance and the theory of heuristic search; the
  relaxed-problem and spanning-tree bounds used here are from this tradition.

Full brief: [BRIEF.md](BRIEF.md).
