# Status

Honest accounting of what runs, what is stubbed, and what is missing. The portfolio's
whole thesis is that the measurements are real, so this file is part of the deliverable.

**Last verified:** `make test` -> 63 tests, OK, 61 s. `make results` -> 5 CSVs
regenerated, 21 min 48 s, of which `eval/regret.py` alone is 19 min: it runs a 12-gap x
5-seed minimax envelope to T = 64,000 and a 1.2M-query fleet horizon. Re-running it
reproduces all five CSVs byte for byte (md5 verified), which is what "every random draw
goes through an explicit seeded generator" is supposed to buy. Python 3.12.3,
numpy 1.26.4, PyYAML 6.0.1, no other dependency.

**The one thing to know before reading any number in this repo:** there are no models
here. No Ollama, no llama.cpp, no vLLM, no weights, no GPU, no network. Every result is
produced against the simulated fleet in `src/fleet/simulator.py` and is labelled
simulated-fleet wherever it appears. That is a deliberate choice of instrument, argued
in the README and in the simulator's module docstring -- the oracle policy is computable,
so regret is measured rather than estimated -- but it is a limitation as well as a
choice, and the rows below say exactly where it bites.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| Simulated fleet: logistic capability ladder, correlated success, cost model | `src/fleet/simulator.py` | complete, tested |
| Exact oracles (expected-reward and hindsight cheapest-sufficient) | `src/fleet/simulator.py` | complete, tested |
| Cost model behind one interface, incl. the `$/token` swap-in | `src/cost.py` | complete, tested |
| Serving-time routing features, with the availability audit | `src/features.py` | complete, tested |
| Baselines: always-small / always-large / random / difficulty-threshold | `src/routers/baselines.py` | complete, tested |
| LinUCB with Sherman-Morrison rank-one updates | `src/routers/linucb.py` | complete, tested |
| Linear Thompson sampling, Gaussian posterior | `src/routers/thompson.py` | complete, tested |
| Budgeted single-price router + online dual update | `src/routers/budgeted.py` | complete, tested |
| Offline multiple-choice knapsack DP + single-price sweep (reference) | `src/routers/budgeted.py` | complete, tested |
| Split conformal calibration, quantile, prediction sets | `src/conformal/calibrate.py` | complete, tested |
| Set-size deferral, multi-tier composition, union bound | `src/conformal/cascade.py` | complete, tested |
| Multi-step agent loop with three real local tools | `src/agent/loop.py`, `src/agent/tools.py` | complete, tested |
| Minimax regret envelope + log-log slope fit | `eval/regret.py` | runs, CSV committed |
| Coverage sweep, cascade composition, exchangeability break | `eval/coverage.py` | runs, CSV committed |
| Cost-quality Pareto table + budget trace | `eval/pareto.py` | runs, CSV committed |
| Compounding: p^n vs measured, correlated and independent | `eval/compounding.py` | runs, CSV committed |
| Surrogate-reward bias study | `eval/surrogate_bias.py` | runs, CSV committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed. Exits with a message naming the package rather than failing obscurely. Every number it would draw is already in `results/*.csv`. |
| `src/fleet/client.py` (`OllamaClient`) | needs a running Ollama daemon, pulled open-weight checkpoints (tens of GB), and network access to fetch them. `generate()` raises `NotImplementedError` naming each of those; the test suite asserts that the message names them. The per-call timing helper is real code and would be reused verbatim. |
| `serve/fleet.yaml` | parses (PyYAML is installed) and is read by the client, but the model names in it are deliberately generic placeholders. Pinning a checkpoint this repo never ran would be inventing a measurement. |

## Not built, and absent rather than mocked

These are the parts of [BRIEF.md](BRIEF.md) that need models, hardware or human labels:

- **Real open-weight models behind the router.** Three arms served by llama.cpp / Ollama
  / vLLM. Nothing here has ever called one.
- **Real GPU-seconds and real peak VRAM.** Every cost in `results/` is a simulated
  wall-clock second from `src/fleet/simulator.py`. The brief's "Peak VRAM" column is
  filled with the simulator's declared per-arm resident memory, which is a parameter,
  not a measurement.
- **A real agent benchmark.** No GAIA subset, no tau-bench, no tool-use suite from
  anywhere but `src/agent/loop.py`'s own generated tasks. The agent loop is real and the
  tools really execute; the *task distribution* is mine and synthetic.
- **LLM-as-judge, and its agreement rate with human labels.** BRIEF.md is explicit that
  a judge is only usable online if you report its agreement with humans on a sample.
  There are no humans, no labels and no judge here, so that row is empty. What
  `eval/surrogate_bias.py` does instead is the complementary half: it shows what a biased
  reward signal *does* to a converging bandit, which is the reason the agreement number
  would matter.
- **A hosted-fleet `$/token` comparison.** `src/cost.py` has the class and the interface;
  it has no price list and is used to produce nothing.
- **Escalation on a real cascade of real models.** The conformal machinery is exercised
  on the simulator's emitted answer distributions. Real models' token-level
  log-likelihoods have different tails, and the mean set sizes reported here would not
  transfer.
- **A non-stationary workload.** Everything here is drawn i.i.d. from a fixed
  distribution. That matters because it is the one axis on which the contextual bandit
  should beat the frozen difficulty-threshold baseline that currently matches it: a
  frozen rule's cut-points are wrong for ever once the workload drifts, while the
  bandit's estimates move. The README claims that as an advantage the bandit *should*
  have; it is not claimed as an advantage that was measured, because it was not.

## What the numbers here do and do not support

**Supported.**

- The minimax regret exponent. LinUCB's supremum-over-instances cumulative regret grows
  with exponent 1/2 on a synthetic linear-reward bandit, and a random policy's with
  exponent 1, under an identical protocol. This is a statement about the algorithm, not
  about any fleet, and it is the repo's load-bearing claim.
- Split conformal's finite-sample coverage, on the simulator's emitted distributions,
  across the whole alpha sweep -- and its collapse under a deliberate distribution shift.
- The union bound composing across cascade tiers, and by how much it is loose.
- The budget constraint being respected by the single-price router and violated by the
  unconstrained one, and the single-price policy matching an offline knapsack DP.
- Sherman-Morrison agreeing with explicit re-inversion.
- The direction and size of the multi-step compounding gap, with an independent-steps
  control that lands on p^n.
- The negative result, which is the one most worth stating plainly: on this workload a
  two-cut-point threshold on a difficulty score matches or beats LinUCB out to 1.2
  million queries, and LinUCB's fleet regret exponent climbs from 0.786 to 0.890 as the
  horizon grows rather than settling -- the signature of a misspecified reward model. The
  bandit does not earn its complexity here, and BRIEF.md asks for that to be published if
  it is true.

**Not supported.** Any claim about a specific open-weight model; any claim about real
latency, real VRAM or real throughput; any claim about task success on a published
agent benchmark; any claim that these routing gains would transfer at these magnitudes
to a real fleet. The *mechanisms* are measured. The *magnitudes* are properties of a
simulator whose parameters I chose, and I chose them to make the routing problem
non-degenerate -- which is stated in the simulator and asserted in the tests, but is
still a choice I made and not a fact about the world.

## A note on layout

CONVENTIONS.md puts the number-producing scripts in `analysis/`. This repo follows
[BRIEF.md](BRIEF.md)'s own suggested layout instead, which names `eval/` for them, and
keeps `analysis/` for the single matplotlib-only plotting script. `make results` runs
`eval/`; `make plots` runs `analysis/`.
