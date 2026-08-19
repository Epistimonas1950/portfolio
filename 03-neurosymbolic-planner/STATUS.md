# Status

Honest accounting of what runs, what is stubbed, and what is missing. The portfolio's
whole thesis is that the measurements are real, so this file is part of the deliverable.

**Last verified:** `make test` → 42 tests, OK, ~5 s. `make results` → 5 CSVs regenerated,
~4 min. Python 3.12.3, numpy 1.26.4, no other dependency, no network.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| State / transitions / cost model | `src/domain.py` | complete, tested |
| Plan validator (capacity, windows, completeness) | `src/domain.py` | complete, tested |
| Seeded instance generator, feasible by construction, horizon knob | `src/domain.py` | complete, tested |
| Exhaustive brute force (ground truth) | `src/search/bruteforce.py` | complete, tested |
| A* with pluggable heuristic + full instrumentation | `src/search/astar.py` | complete, tested |
| IDA* | `src/search/idastar.py` | complete, tested |
| Heuristics h0 / h1 / h2 + a deliberately inconsistent one | `src/search/heuristics.py` | complete, proofs in docstrings, tested |
| Minimax, alpha–beta, move ordering, transposition table, iterative deepening | `src/search/alphabeta.py` | complete, tested |
| Effective-branching-factor fit, node counters | `src/search/instrument.py` | complete |
| NL → JSON, offline rule backend | `src/translate.py` | complete, tested, benchmarked |
| Plan → English, template based | `src/explain.py` | complete, tested |
| CLI end-to-end demo | `src/demo.py` | complete, `make demo` |
| Hand-written NL corpus, 31 requests + gold | `data/nl_requests.json` | complete |
| Stop-name gazetteer | `data/gazetteer.json` | complete |
| Optimality vs horizon + the node-budget wall | `bench/optimality.py` | runs, CSV committed |
| Heuristic dominance | `bench/dominance.py` | runs, CSV committed |
| Minimax vs alpha–beta vs ordering vs TT | `bench/adversarial.py` | runs, CSV committed |
| Translation accuracy + failure taxonomy | `bench/translation_accuracy.py` | runs, 2 CSVs committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed on this machine. Exits with a message naming the package rather than failing obscurely. Every number it draws is already in `results/*.csv`. One panel of the headline figure is drawn deliberately empty — see below. |
| `src/translate.OllamaBackend` | raises `NotImplementedError` naming the model, the server and the constrained-decoding step it needs. It does **not** fall back to `RuleBackend`, because a silent fallback would make every number attributed to the language-model path a measurement of the rule parser instead. |

## Not built, and absent rather than mocked

- **The pure-LLM chain-of-thought baseline.** BRIEF.md's table B and its headline plot
  compare A* optimality against a language model's on the identical instances. There is
  no local model, no GPU and no network on this machine, so **that curve has not been
  measured and the row is empty.** `analysis/plot_results.py` draws the axis with the
  note in place rather than a plausible line. Everything needed to run it exists — the
  instances are seeded and reproducible, the optimum is known for every one of them, and
  `translate.OllamaBackend` is the socket it plugs into — so this is a missing
  measurement, not a missing design.
- **The Gradio demo.** Gradio is not installed and there is no network. `make demo` is
  the same pipeline behind a command line, and it runs today. The web front end would add
  presentation, not capability.
- **A real LLM translation backend.** The accuracy number in the README is the *rule*
  parser's. A language model would very likely handle several of the failure modes the
  rule parser cannot (relative times, coreference, arithmetic) and would introduce its
  own — most importantly, wrong-and-confident parses, of which the rule parser produced
  zero. The comparison is the obvious next experiment and it is not in this repo.
- **LP-relaxation and pattern-database heuristics.** BRIEF.md lists three sources of
  admissible bounds. Two are here (a relaxed problem in two strengths; the spanning-tree
  bound). The LP relaxation would need an LP solver, which would mean scipy, which is not
  installed. A pattern database is buildable with what is here and is simply not done.
- **A performance core in Rust or C++.** The search is pure Python. The wall in
  `results/optimality.csv` is a Python wall as much as an algorithmic one, and the README
  does not claim otherwise.

## What the numbers here do and do not support

**Supported.** A* returns the exact exhaustive optimum on all 240 verified runs (60
instances, 3–7 stops, 4 heuristics); expansions fall monotonically with heuristic
strength, by 6.29× from h0 to h2 at 9 stops, with zero violations of the dominance
theorem over 60 instance-pairs; consistent heuristics produce exactly zero re-expansions
and the admissible-but-inconsistent control produces 1 157 at 9 stops while staying
optimal; alpha–beta returns the identical minimax value while expanding 267.5× fewer
nodes at 8 plies with ordering and a transposition table; the rule parser is exact on
100% of in-grammar requests and 77.4% of the whole corpus, and flagged every request it
got wrong.

**Not supported.** Any claim about a language model's planning accuracy — none was run.
Any claim that 77.4% generalises to real dispatcher traffic — the corpus is self-authored
and that bounds what the number can mean. Any claim about wall-clock performance of a
compiled implementation. Any claim that the approach scales past the measured ceiling of
22 stops.
